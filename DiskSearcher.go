package main

import (
	"bufio"
	"bytes"
	"errors" // Added for custom errors like errInterrupted
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	// No need for context, syscall, or os/signal here if we stick to the 'done' channel pattern
)

// --- Constants for General Search ---
const (
	defaultWriterBufferSize = 128 * 1024 * 1024 // 128 MB buffer for output writer (Increased from 8MB)
	consoleMatchLimit       = 10                // Max matches to show on console
	progressUpdateInterval  = 500 * time.Millisecond
	defaultMaxFileSizeMB    = 150
	defaultWorkers          = 4  // Default workers remains 4
	defaultBufferSizeMB     = 16 // Increased default per-worker buffer
)

// --- Constants for TikTok Mode ---
const (
	tiktokTargetDomain      = "tiktok.com"
	tiktokCookieName1       = "sid_guard"
	tiktokCookieNameIndex1  = 5 // Index of name field in Netscape cookie format
	tiktokCookieValueIndex1 = 6 // Index of value field
	tiktokCookieName2       = "perf_feed_cache"
	// tiktokCookieValueIndex2 = 6 // Assuming same value index
	tiktokMinFields         = 7               // Minimum fields for a valid cookie line
	tiktokScannerBufferSize = 2 * 1024 * 1024 // 2 MB buffer specific to TikTok scanner if needed, or use main buffer
)

// Custom error for interruption (useful if we reintroduce signal handling later)
// var errInterrupted = errors.New("process interrupted by user")

// --- Pre-compiled Regular Expressions ---
var (
	emailPassRegex = regexp.MustCompile(`([^:\s]+@[^:\s]+\.[^:\s]+):([^:\s]+)`)
)

// --- Configuration parameters ---
type SearchConfig struct {
	FolderPath           string
	SearchTerm           string // Required for non-TikTok modes
	OutputPath           string
	MaxOccurrences       int64
	StopAfterMax         bool
	BufferSizeMB         int
	Workers              int
	Verbose              bool
	Mode                 string // "email", "url", "user", or "tiktok"
	ShowFilePath         bool
	MaxFileSizeBytes     int64
	LowerSearchTerm      string // Only used for non-TikTok modes
	LowerSearchTermBytes []byte // Only used for non-TikTok modes
	DebugMode            bool   // Added debug flag
}

// --- Statistics for reporting ---
type SearchStats struct {
	FilesFound         atomic.Int64
	FilesProcessed     atomic.Int64
	FilesSkipped       atomic.Int64
	BytesProcessed     atomic.Int64
	MatchesFound       atomic.Int64
	DuplicatesSkipped  atomic.Int64
	NonMatchingSkipped atomic.Int64 // Count lines matching term but not format (non-tiktok modes)
	ErrorsEncountered  atomic.Int64
	StartTime          time.Time
	mu                 sync.Mutex
}

// printProgress (remains the same)
func (s *SearchStats) printProgress() {
	s.mu.Lock()
	defer s.mu.Unlock()
	elapsed := time.Since(s.StartTime).Seconds()
	mbProcessed := float64(s.BytesProcessed.Load()) / (1024 * 1024)
	mbPerSecond := 0.0
	if elapsed > 0 {
		mbPerSecond = mbProcessed / elapsed
	}
	fmt.Printf("\rProgress: %d files found, %d processed, %d skipped, %d errors, %d unique matches (%.2f MB/s)  ",
		s.FilesFound.Load(), s.FilesProcessed.Load(), s.FilesSkipped.Load(), s.ErrorsEncountered.Load(), s.MatchesFound.Load(), mbPerSecond)
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "FATAL ERROR: Program crashed: %v\n", r)
			fmt.Fprintln(os.Stderr, "Please report this error with the conditions that caused it.")
			os.Exit(1)
		}
	}()

	runtime.GOMAXPROCS(0)

	// --- Flag Definition ---
	folderPath := flag.String("folder", "F:\\cloud_split", "Input folder path to search")
	searchTerm := flag.String("term", "", "Text to search for (required for email/url/user modes, case-insensitive)")
	outputPath := flag.String("output", "", "Output file path (optional, generates '<term|mode>_<timestamp>.txt' if empty)")
	maxOccurrences := flag.Int64("max", 0, "Stop after finding this many unique occurrences (0 = unlimited)")
	stopAfterMax := flag.Bool("stop-after-max", false, "Stop searching all files once 'max' unique occurrences are found")
	bufferSizeMB := flag.Int("buffer", defaultBufferSizeMB, "Per-worker buffer size in MB for file reading")
	workers := flag.Int("workers", defaultWorkers, "Number of concurrent worker goroutines")
	verbose := flag.Bool("v", false, "Enable verbose logging (placeholder)")
	showFilePath := flag.Bool("show-path", false, "Include file path ([filepath:line/0]) in output lines")
	maxFileSize := flag.Int64("max-file-size", defaultMaxFileSizeMB, "Maximum file size in MB to process (0 = unlimited)")
	debugMode := flag.Bool("debug", false, "Enable detailed debug logging for finding pairs (TikTok mode)") // Debug flag

	// Mode flags
	email := flag.Bool("email", false, "Extract 'email:password' format") // Default is now false
	url := flag.Bool("url", false, "Output the full matching line")
	user := flag.Bool("user", false, "Extract 'username:password' format")
	tiktok := flag.Bool("tiktok", false, "Extract TikTok 'sid_guard|perf_feed_cache' cookie pairs") // NEW MODE

	flag.Parse()

	// --- Determine Mode and Validate Inputs ---
	mode := ""
	modeCount := 0
	outputNameHint := *searchTerm // Use search term for output name by default

	if *email {
		mode = "email"
		modeCount++
	}
	if *url {
		mode = "url"
		modeCount++
	}
	if *user {
		mode = "user"
		modeCount++
	}
	if *tiktok {
		mode = "tiktok"
		modeCount++
		outputNameHint = "tiktok_cookies" // Use specific name for tiktok mode output
	}

	if modeCount == 0 {
		fmt.Fprintln(os.Stderr, "Error: A mode must be specified (-email, -url, -user, or -tiktok).")
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		os.Exit(1)
	}
	if modeCount > 1 {
		fmt.Fprintln(os.Stderr, "Error: Only one mode (-email, -url, -user, -tiktok) can be specified at a time.")
		os.Exit(1)
	}

	// Validate search term requirement
	if mode != "tiktok" && *searchTerm == "" {
		fmt.Fprintf(os.Stderr, "Error: -term parameter is required for '%s' mode.\n", mode)
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		os.Exit(1)
	}
	if mode == "tiktok" && *searchTerm != "" {
		fmt.Fprintln(os.Stderr, "Warning: -term parameter is ignored when using -tiktok mode.")
	}

	// Validate other parameters
	if *bufferSizeMB <= 0 {
		fmt.Fprintf(os.Stderr, "Warning: Invalid buffer size (%d MB), using default %d MB\n", *bufferSizeMB, defaultBufferSizeMB)
		*bufferSizeMB = defaultBufferSizeMB
	}
	if *workers <= 0 {
		fmt.Fprintf(os.Stderr, "Warning: Invalid worker count (%d), using default %d workers\n", *workers, defaultWorkers)
		*workers = defaultWorkers
	}
	if *maxFileSize < 0 {
		fmt.Fprintln(os.Stderr, "Warning: Invalid max file size, allowing all sizes (0)")
		*maxFileSize = 0
	}

	// --- Output Path Setup ---
	if *outputPath == "" {
		successesDir := filepath.Join("C:\\Users\\admin\\Documents\\searchHDD")
		if err := os.MkdirAll(successesDir, 0755); err != nil {
			fmt.Fprintf(os.Stderr, "Error creating output directory '%s': %v\n", successesDir, err)
			os.Exit(1)
		}
		safeHint := strings.Map(func(r rune) rune {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
				return r
			}
			return '_'
		}, outputNameHint) // Use outputNameHint here
		if len(safeHint) > 30 {
			safeHint = safeHint[:30]
		}
		timeStamp := time.Now().Format("20060102_150405")
		*outputPath = filepath.Join(successesDir, fmt.Sprintf("%s_%s.txt", safeHint, timeStamp))
	}

	// --- Configuration Creation ---
	maxFileSizeBytes := int64(0)
	if *maxFileSize > 0 {
		maxFileSizeBytes = *maxFileSize * 1024 * 1024
	}

	var lowerSearchTermStr string
	var lowerSearchTermBytes []byte
	if mode != "tiktok" {
		lowerSearchTermStr = strings.ToLower(*searchTerm)
		lowerSearchTermBytes = []byte(lowerSearchTermStr)
	}

	config := SearchConfig{
		FolderPath:           *folderPath,
		SearchTerm:           *searchTerm, // Store original term even if unused by tiktok
		LowerSearchTerm:      lowerSearchTermStr,
		LowerSearchTermBytes: lowerSearchTermBytes,
		OutputPath:           *outputPath,
		MaxOccurrences:       *maxOccurrences,
		StopAfterMax:         *stopAfterMax,
		BufferSizeMB:         *bufferSizeMB,
		Workers:              *workers,
		Verbose:              *verbose,
		Mode:                 mode,
		ShowFilePath:         *showFilePath,
		MaxFileSizeBytes:     maxFileSizeBytes,
		DebugMode:            *debugMode, // Pass debug flag
	}

	// --- Start Search ---
	fmt.Println("Starting high-performance file search...")
	fmt.Printf("  Folder:        %s\n", config.FolderPath)
	if mode != "tiktok" {
		fmt.Printf("  Search Term:   '%s' (case-insensitive)\n", config.SearchTerm)
	} else {
		fmt.Printf("  Search Target: TikTok cookie pairs ('%s', '%s')\n", tiktokCookieName1, tiktokCookieName2)
	}
	fmt.Printf("  Output File:   %s\n", config.OutputPath)
	fmt.Printf("  Mode:          %s\n", config.Mode)
	fmt.Printf("  Max File Size: %d MB\n", *maxFileSize)
	fmt.Printf("  Workers:       %d\n", config.Workers)
	fmt.Printf("  Buffer Size:   %d MB/worker\n", config.BufferSizeMB)
	fmt.Printf("  Show Path:     %t\n", config.ShowFilePath)
	if config.DebugMode && config.Mode == "tiktok" {
		fmt.Println("  Debug Logging: Enabled for TikTok pair finding")
	}
	if config.MaxOccurrences > 0 {
		fmt.Printf("  Max Occurrences: %d%s\n", config.MaxOccurrences, map[bool]string{true: " (will stop early)", false: ""}[config.StopAfterMax])
	}

	matches, err := searchFiles(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError during search: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nSearch completed. Found %d unique matches.\n", matches)
	fmt.Printf("Results saved to: %s\n", config.OutputPath)
}

// processLine (remains the same, handles email/url/user modes)
func processLine(line string, config SearchConfig) string {
	// ... (implementation is identical to the one you provided)
	switch config.Mode {
	case "url":
		return line
	case "email":
		if !strings.ContainsRune(line, '@') {
			return ""
		}
		matches := emailPassRegex.FindStringSubmatch(line)
		if len(matches) >= 3 {
			return matches[1] + ":" + matches[2]
		}
		if strings.ContainsRune(line, '|') {
			parts := strings.Split(line, "|")
			for i, part := range parts {
				trimmedPart := strings.TrimSpace(part)
				if strings.ContainsRune(trimmedPart, '@') && i+1 < len(parts) {
					return trimmedPart + ":" + strings.TrimSpace(parts[i+1])
				}
			}
		}
		parts := strings.Split(line, ":")
		if len(parts) >= 2 {
			for i, part := range parts {
				trimmedPart := strings.TrimSpace(part)
				if strings.ContainsRune(trimmedPart, '@') && strings.ContainsRune(trimmedPart, '.') && i+1 < len(parts) {
					// Consider joining remaining parts if password contains ':'
					// return trimmedPart + ":" + strings.TrimSpace(strings.Join(parts[i+1:], ":"))
					return trimmedPart + ":" + strings.TrimSpace(parts[i+1])
				}
			}
		}
		return ""

	case "user":
		parts := strings.Split(line, ":")
		if len(parts) >= 3 {
			// return strings.TrimSpace(parts[1]) + ":" + strings.TrimSpace(strings.Join(parts[2:], ":"))
			return strings.TrimSpace(parts[1]) + ":" + strings.TrimSpace(parts[2])
		}
		if strings.ContainsRune(line, '|') {
			parts = strings.Split(line, "|")
			if len(parts) >= 3 {
				// return strings.TrimSpace(parts[1]) + ":" + strings.TrimSpace(strings.Join(parts[2:], "|"))
				return strings.TrimSpace(parts[1]) + ":" + strings.TrimSpace(parts[2])
			}
		}
		return ""
	// NOTE: "tiktok" mode is handled BEFORE calling this function in the writer goroutine.
	default:
		return ""
	}
}

// searchFiles orchestrates the file search process.
func searchFiles(config SearchConfig) (int64, error) {
	outFile, err := os.Create(config.OutputPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create output file '%s': %v", config.OutputPath, err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, defaultWriterBufferSize)
	defer func() {
		if ferr := writer.Flush(); ferr != nil {
			fmt.Fprintf(os.Stderr, "\nWarning: Error flushing output file buffer: %v\n", ferr)
		}
	}()

	var wg sync.WaitGroup
	fileChan := make(chan string, config.Workers*2)
	// resultChan carries strings in format "[filepath:lineNum|0] content"
	// where content is the raw line for url/email/user, or value1|value2 for tiktok
	resultChan := make(chan string, 100)
	done := make(chan struct{})
	var closeDoneOnce sync.Once

	stats := &SearchStats{StartTime: time.Now()}

	var progressWg sync.WaitGroup
	progressWg.Add(1)
	go func() {
		defer progressWg.Done()
		ticker := time.NewTicker(progressUpdateInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				stats.printProgress()
			case <-done:
				stats.printProgress()
				fmt.Println()
				return
			}
		}
	}()

	var writerWg sync.WaitGroup
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		seen := make(map[string]struct{})
		var consoleCount int
		var duplicatesFound int64 // Track duplicates specific to writer

		for result := range resultChan {
			// Parse the standardized result format: "[filepath:lineNum|0] content"
			fileInfo, content, parseOK := parseResultString(result)
			if !parseOK {
				if config.DebugMode { // Optionally log bad parse
					fmt.Fprintf(os.Stderr, "\nDEBUG: [Writer] Failed to parse result string: %q\n", result)
				}
				stats.ErrorsEncountered.Add(1)
				continue
			}

			var outputLine string
			var isTikTokPair bool = (config.Mode == "tiktok")

			if isTikTokPair {
				// For TikTok mode, the 'content' is already the desired "value1|value2"
				outputLine = content
				if outputLine == "" { // Should not happen if processFileTikTok sends valid pairs
					continue
				}
			} else {
				// For other modes, process the raw line content
				outputLine = processLine(content, config)
				if outputLine == "" {
					stats.NonMatchingSkipped.Add(1)
					continue
				}
				// For URL mode, processLine returns the original content
			}

			// --- Deduplication and Writing (Mutex Protected) ---
			stats.mu.Lock() // Lock for map access and potential console output

			if _, isDuplicate := seen[outputLine]; !isDuplicate {
				seen[outputLine] = struct{}{} // Mark as seen

				// Construct final output line for the file
				var fullOutputLine string
				if config.ShowFilePath {
					fullOutputLine = fileInfo + " " + outputLine // fileInfo includes [filepath:line/0]
				} else {
					fullOutputLine = outputLine
				}

				// Write to buffered output file
				if _, werr := fmt.Fprintln(writer, fullOutputLine); werr != nil {
					stats.ErrorsEncountered.Add(1)
					fmt.Fprintf(os.Stderr, "\nError writing to output file: %v\n", werr)
					_ = writer.Flush()
					// Unlock before returning/continuing on error
					stats.mu.Unlock()
					continue // Skip further processing for this line
				}

				// Immediate flush
				if ferr := writer.Flush(); ferr != nil {
					stats.ErrorsEncountered.Add(1)
					fmt.Fprintf(os.Stderr, "\nError flushing output file: %v\n", ferr)
				}

				currentMatches := stats.MatchesFound.Add(1)

				// Show limited matches on console
				if consoleCount < consoleMatchLimit {
					// Show processed line + file info if requested for clarity
					displayLine := outputLine
					if config.ShowFilePath {
						displayLine = fileInfo + " " + outputLine
					}
					// Add a newline before the first "Found:" to separate from progress line
					if consoleCount == 0 {
						fmt.Println()
					}
					fmt.Println("Found:", displayLine)
					consoleCount++
					if consoleCount == consoleMatchLimit {
						fmt.Println("(Further matches will be written to file only. Press Ctrl+C to stop)")
					}
				}
				stats.mu.Unlock() // Unlock after successful processing

				// Check max occurrences (outside mutex to avoid holding lock)
				if config.StopAfterMax && config.MaxOccurrences > 0 && currentMatches >= config.MaxOccurrences {
					closeDoneOnce.Do(func() { close(done) })
				}

			} else {
				// Duplicate found
				duplicatesFound++ // Use local counter or atomic stats.DuplicatesSkipped
				stats.DuplicatesSkipped.Add(1)
				stats.mu.Unlock()                     // Unlock if it was a duplicate
				if config.DebugMode && isTikTokPair { // Debug duplicate TikTok pairs
					fmt.Printf("DEBUG: [Writer] Skipping duplicate TikTok pair: %s (from %s)\n", outputLine, fileInfo)
				}
			}
		}
		// Final message from writer if needed (or rely on main)
		// fmt.Printf("[Writer] Finished. Processed results. Duplicates skipped by writer: %d\n", duplicatesFound)
	}()

	// --- Worker Goroutines ---
	workerBufferSize := config.BufferSizeMB * 1024 * 1024
	for i := 0; i < config.Workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			// Allocate buffer per worker. Reuse for multiple files.
			buf := make([]byte, workerBufferSize)

			for {
				select {
				case filePath, ok := <-fileChan:
					if !ok {
						return // fileChan closed
					}

					// --- Select appropriate processing function based on mode ---
					switch config.Mode {
					case "tiktok":
						processFileTikTok(filePath, config, resultChan, buf, stats, done)
					case "email", "url", "user":
						processFile(filePath, config, resultChan, buf, stats, done)
					default:
						// Should not happen due to upfront validation
						fmt.Fprintf(os.Stderr, "\nError: Unknown mode '%s' in worker %d\n", config.Mode, workerID)
						stats.FilesSkipped.Add(1) // Count as skipped
					}
					// -------------------------------------------------------------

				case <-done:
					return // Stop signal
				}
			}
		}(i)
	}

	// --- File Finder Goroutine ---
	var finderWg sync.WaitGroup
	finderWg.Add(1)
	go func() {
		defer finderWg.Done()
		defer close(fileChan) // Close fileChan when walk completes or stops

		allowedExts := map[string]struct{}{ // Default extensions
			".txt": {}, ".log": {}, ".csv": {}, ".json": {}, ".xml": {}, ".html": {}, ".ini": {}, ".conf": {}, ".yaml": {}, ".yml": {},
		}
		// For TikTok mode, we ONLY care about .txt files
		if config.Mode == "tiktok" {
			allowedExts = map[string]struct{}{".txt": {}}
		}

		err := filepath.WalkDir(config.FolderPath, func(path string, d fs.DirEntry, walkErr error) error {
			select {
			case <-done:
				return fs.SkipAll
			default:
			}

			if walkErr != nil {
				stats.ErrorsEncountered.Add(1)
				// Don't print every permission error to avoid spam
				// fmt.Fprintf(os.Stderr, "\nWarning: Error accessing path %s: %v\n", path, walkErr)
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}

			if !d.IsDir() {
				// Check extension based on allowed map
				ext := strings.ToLower(filepath.Ext(path))
				if _, ok := allowedExts[ext]; ok {
					stats.FilesFound.Add(1)
					select {
					case fileChan <- path:
					case <-done:
						return fs.SkipAll
					}
				} else {
					// Optionally count skipped files by extension type if needed
				}
			}
			return nil
		})
		if err != nil && !errors.Is(err, fs.SkipAll) {
			fmt.Fprintf(os.Stderr, "\nWarning: File traversal error: %v\n", err)
			stats.ErrorsEncountered.Add(1)
		}
	}()

	// --- Wait for Completion ---
	finderWg.Wait()
	wg.Wait()
	close(resultChan)
	writerWg.Wait()
	closeDoneOnce.Do(func() { close(done) }) // Ensure done is closed if not already
	progressWg.Wait()

	return stats.MatchesFound.Load(), nil
}

// parseResultString helps standardize parsing the worker output
func parseResultString(result string) (fileInfo, content string, ok bool) {
	// Expects format: "[filepath:lineNum|0] content"
	// Find the closing bracket and the space after it
	idx := strings.Index(result, "] ")
	if idx == -1 || idx+2 >= len(result) {
		return "", "", false // Invalid format
	}
	fileInfo = result[:idx+1] // Include the bracket: "[filepath:lineNum|0]"
	content = result[idx+2:]  // The rest is content
	ok = true
	return
}

// processFile (Handles email/url/user modes - remains mostly the same as your version)
func processFile(filePath string, config SearchConfig, resultChan chan<- string, buf []byte, stats *SearchStats, done <-chan struct{}) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "\nPANIC recovered while processing file %s: %v\n", filePath, r)
			stats.ErrorsEncountered.Add(1)
		}
	}()

	info, err := os.Stat(filePath)
	if err != nil {
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1) // Count stat errors
		return
	}
	fileSize := info.Size()
	if fileSize == 0 || (config.MaxFileSizeBytes > 0 && fileSize > config.MaxFileSizeBytes) {
		stats.FilesSkipped.Add(1)
		return
	}

	file, err := os.Open(filePath)
	if err != nil {
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1) // Count open errors
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(buf, len(buf)) // Use provided buffer

	lineNum := 0
	bytesReadSinceCheck := int64(0) // Track bytes for periodic 'done' check

	lowerSearchBytes := config.LowerSearchTermBytes

	for scanner.Scan() {
		lineNum++
		lineBytes := scanner.Bytes()
		lineLen := int64(len(lineBytes))
		bytesReadSinceCheck += lineLen + 1 // +1 for newline

		// Optimization: lowercase and check using bytes.Contains
		if bytes.Contains(bytes.ToLower(lineBytes), lowerSearchBytes) {
			lineStr := string(lineBytes) // Allocate string only on match
			// Format result consistently: [filepath:lineNum] content
			result := fmt.Sprintf("[%s:%d] %s", filePath, lineNum, lineStr)
			select {
			case resultChan <- result:
				// Sent
			case <-done:
				stats.BytesProcessed.Add(bytesReadSinceCheck) // Add bytes processed before stopping
				stats.FilesProcessed.Add(1)                   // Mark as processed even if stopped mid-way
				return
			}
		}

		// Check 'done' channel periodically to avoid blocking on huge files
		// Check based on bytes read for better granularity than line count
		if bytesReadSinceCheck > 1*1024*1024 { // Check every ~1MB
			select {
			case <-done:
				stats.BytesProcessed.Add(bytesReadSinceCheck)
				stats.FilesProcessed.Add(1)
				return
			default:
				// Continue
			}
			bytesReadSinceCheck = 0 // Reset counter
		}
	}

	// Add remaining bytes for the file
	stats.BytesProcessed.Add(bytesReadSinceCheck)

	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			fmt.Fprintf(os.Stderr, "\nWarning: Line too long in file %s (max buffer: %d MB).\n", filePath, config.BufferSizeMB)
		} else {
			fmt.Fprintf(os.Stderr, "\nWarning: Error scanning file %s: %v\n", filePath, err)
		}
		stats.ErrorsEncountered.Add(1)
		stats.FilesSkipped.Add(1) // Mark as skipped due to scanner error
	} else {
		stats.FilesProcessed.Add(1) // Processed fully
	}
}

// --- NEW: processFileTikTok ---
// Handles searching for TikTok cookie pairs within a single file.
func processFileTikTok(filePath string, config SearchConfig, resultChan chan<- string, buf []byte, stats *SearchStats, done <-chan struct{}) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "\nPANIC recovered while processing file %s (TikTok mode): %v\n", filePath, r)
			stats.ErrorsEncountered.Add(1)
		}
	}()

	info, err := os.Stat(filePath)
	if err != nil {
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1)
		return
	}
	fileSize := info.Size()
	if fileSize == 0 || (config.MaxFileSizeBytes > 0 && fileSize > config.MaxFileSizeBytes) {
		stats.FilesSkipped.Add(1)
		return
	}

	file, err := os.Open(filePath)
	if err != nil {
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Use the main buffer OR the specific TikTok buffer size if defined differently
	scanner.Buffer(buf, len(buf)) // Using the per-worker buffer passed in

	var foundVal1, foundVal2 string
	lineNumber := 0
	bytesReadSinceCheck := int64(0)

	for scanner.Scan() {
		lineNumber++
		line := scanner.Text() // Get as string for splitting
		lineLen := int64(len(line))
		bytesReadSinceCheck += lineLen + 1

		// --- TikTok Specific Logic ---
		// Only process if we haven't found both values yet
		if foundVal1 == "" || foundVal2 == "" {
			fields := strings.Split(line, "\t") // Split by tab

			if len(fields) < tiktokMinFields || !strings.Contains(fields[0], tiktokTargetDomain) {
				continue // Skip lines not matching basic criteria
			}

			// Check indices *before* accessing them
			if len(fields) <= tiktokCookieNameIndex1 || len(fields) <= tiktokCookieValueIndex1 {
				continue
			}

			cookieName := fields[tiktokCookieNameIndex1]
			cookieValue := fields[tiktokCookieValueIndex1]

			if foundVal1 == "" && cookieName == tiktokCookieName1 {
				foundVal1 = cookieValue
				if config.DebugMode {
					fmt.Printf("DEBUG: Found '%s' in '%s' (Line %d)\n", tiktokCookieName1, filePath, lineNumber)
				}
			}
			if foundVal2 == "" && cookieName == tiktokCookieName2 {
				foundVal2 = cookieValue
				if config.DebugMode {
					fmt.Printf("DEBUG: Found '%s' in '%s' (Line %d)\n", tiktokCookieName2, filePath, lineNumber)
				}
			}
		}
		// --- End TikTok Logic ---

		// Check if BOTH cookies have been found in this file
		if foundVal1 != "" && foundVal2 != "" {
			if config.DebugMode {
				fmt.Printf("DEBUG: Found BOTH '%s' and '%s' in '%s'. Sending result.\n", tiktokCookieName1, tiktokCookieName2, filePath)
			}
			// Format the result: value1|value2
			pairResult := fmt.Sprintf("%s|%s", foundVal1, foundVal2)
			// Standardize output format: [filepath:0] content (use 0 for lineNum)
			resultString := fmt.Sprintf("[%s:0] %s", filePath, pairResult)

			select {
			case resultChan <- resultString:
				// Sent the pair, we are done with this file for TikTok mode
				stats.BytesProcessed.Add(fileSize) // Add full size since we found the pair
				stats.FilesProcessed.Add(1)
				return // Exit function for this file
			case <-done:
				// Stop signal received before sending
				stats.BytesProcessed.Add(bytesReadSinceCheck) // Add bytes read so far
				stats.FilesProcessed.Add(1)
				return // Exit function
			}
		}

		// Periodic check for done signal
		if bytesReadSinceCheck > 1*1024*1024 { // Check every ~1MB
			select {
			case <-done:
				stats.BytesProcessed.Add(bytesReadSinceCheck)
				stats.FilesProcessed.Add(1)
				return
			default:
				// Continue
			}
			bytesReadSinceCheck = 0 // Reset counter
		}
	}

	// Add remaining bytes processed if loop finished without finding pair or error
	stats.BytesProcessed.Add(bytesReadSinceCheck)

	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			fmt.Fprintf(os.Stderr, "\nWarning: Line too long in file %s (TikTok mode, max buffer: %d MB).\n", filePath, config.BufferSizeMB)
		} else {
			fmt.Fprintf(os.Stderr, "\nWarning: Error scanning file %s (TikTok mode): %v\n", filePath, err)
		}
		stats.ErrorsEncountered.Add(1)
		stats.FilesSkipped.Add(1) // Skip file due to error
	} else {
		// Reached end of file without finding the pair or scanner error
		stats.FilesProcessed.Add(1)
	}
}
