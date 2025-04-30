package main

import (
	"bufio"
	"bytes"
	"errors"
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
	defaultWriterBufferSize = 128 * 1024 * 1024 // 128 MB buffer for output writer
	consoleMatchLimit       = 10                // Max matches to show on console
	progressUpdateInterval  = 500 * time.Millisecond
	defaultMaxFileSizeMB    = 15000 // Default maximum file size to process
	defaultWorkers          = 4     // Default number of concurrent workers
	defaultBufferSizeMB     = 32    // Default per-worker buffer size for file reading
)

// --- Constants for TikTok Mode ---
const (
	tiktokTargetDomain      = "tiktok.com"      // Domain to look for in cookie files
	tiktokCookieName1       = "sid_guard"       // First target cookie name
	tiktokCookieNameIndex1  = 5                 // Index of name field in Netscape cookie format (0-based)
	tiktokCookieValueIndex1 = 6                 // Index of value field
	tiktokCookieName2       = "perf_feed_cache" // Second target cookie name
	// tiktokCookieValueIndex2 = 6 // Assuming same value index as sid_guard
	tiktokMinFields         = 7               // Minimum fields for a valid Netscape cookie line
	tiktokScannerBufferSize = 2 * 1024 * 1024 // 2 MB buffer specific to TikTok scanner (if different needed, currently uses main buffer)
)

// --- Pre-compiled Regular Expressions ---
var (
	// Regex for email:password extraction (adjust if needed for more formats)
	emailPassRegex = regexp.MustCompile(`([^:\s]+@[^:\s]+\.[^:\s]+):([^:\s]+)`)
)

// --- Configuration parameters ---
type SearchConfig struct {
	FolderPath           string // Directory to search within
	SearchTerm           string // Text to search for (required for non-TikTok modes)
	OutputPath           string // File to write results to
	MaxOccurrences       int64  // Stop after finding this many unique matches (0=unlimited)
	StopAfterMax         bool   // Stop all workers once MaxOccurrences is reached
	BufferSizeMB         int    // Per-worker read buffer size in MB
	Workers              int    // Number of concurrent search workers
	Verbose              bool   // Enable more detailed logging (currently placeholder)
	Mode                 string // "email", "url", "user", or "tiktok"
	ShowFilePath         bool   // Prepend "[filepath:line]" to output lines
	MaxFileSizeBytes     int64  // Maximum file size in bytes to process (0=unlimited)
	LowerSearchTerm      string // Lowercase version of SearchTerm
	LowerSearchTermBytes []byte // Lowercase SearchTerm as byte slice for efficiency
	DebugMode            bool   // Enable detailed debug logging, especially for TikTok pairing
}

// --- Statistics for reporting ---
type SearchStats struct {
	FilesFound         atomic.Int64 // Total files discovered matching criteria (e.g., extension)
	FilesProcessed     atomic.Int64 // Files fully read or processed until stopped
	FilesSkipped       atomic.Int64 // Files skipped (too large, zero size, wrong extension, error)
	BytesProcessed     atomic.Int64 // Total bytes read from processed files
	MatchesFound       atomic.Int64 // Unique matches found and written to output
	DuplicatesSkipped  atomic.Int64 // Duplicate matches found but not written
	NonMatchingSkipped atomic.Int64 // Lines containing search term but not matching format (non-TikTok)
	ErrorsEncountered  atomic.Int64 // Errors during file access, reading, etc.
	StartTime          time.Time    // When the search began
	mu                 sync.Mutex   // Mutex to protect console output during progress updates
}

// printProgress displays the current search statistics to the console.
func (s *SearchStats) printProgress() {
	s.mu.Lock()
	defer s.mu.Unlock()
	elapsed := time.Since(s.StartTime).Seconds()
	mbProcessed := float64(s.BytesProcessed.Load()) / (1024 * 1024)
	mbPerSecond := 0.0
	if elapsed > 0 {
		mbPerSecond = mbProcessed / elapsed
	}
	// Using \r for carriage return to update the line in place
	fmt.Printf("\rProgress: %d files found, %d processed, %d skipped, %d errors, %d unique matches (%.2f MB/s)  ",
		s.FilesFound.Load(), s.FilesProcessed.Load(), s.FilesSkipped.Load(), s.ErrorsEncountered.Load(), s.MatchesFound.Load(), mbPerSecond)
}

func main() {
	// Graceful panic recovery
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "\n\nFATAL ERROR: Program crashed: %v\n", r)
			fmt.Fprintln(os.Stderr, "Please report this error with the conditions that caused it.")
			os.Exit(1) // Exit with non-zero status on crash
		}
	}()

	// Use all available CPU cores
	runtime.GOMAXPROCS(0)

	// --- Flag Definition ---
	// Changed default folder to "." (current directory)
	folderPath := flag.String("folder", ".", "Input folder path to search (default: current directory)")
	searchTerm := flag.String("term", "", "Text to search for (required for email/url/user modes, case-insensitive)")
	// Default output path is now generated in the current directory
	outputPath := flag.String("output", "", "Output file path (optional, generates '<term|mode>_<timestamp>.txt' in current dir if empty)")
	maxOccurrences := flag.Int64("max", 0, "Stop after finding this many unique occurrences (0 = unlimited)")
	stopAfterMax := flag.Bool("stop-after-max", false, "Stop searching all files once 'max' unique occurrences are found")
	bufferSizeMB := flag.Int("buffer", defaultBufferSizeMB, "Per-worker buffer size in MB for file reading")
	workers := flag.Int("workers", defaultWorkers, fmt.Sprintf("Number of concurrent worker goroutines (default: %d, uses %d cores)", defaultWorkers, runtime.NumCPU()))
	verbose := flag.Bool("v", false, "Enable verbose logging (placeholder for future use)")
	showFilePath := flag.Bool("show-path", false, "Include file path ([filepath:line/0]) in output lines")
	maxFileSize := flag.Int64("max-file-size", defaultMaxFileSizeMB, "Maximum file size in MB to process (0 = unlimited)")
	debugMode := flag.Bool("debug", false, "Enable detailed debug logging for finding pairs (TikTok mode)")

	// Mode flags - specify the type of search/extraction
	email := flag.Bool("email", false, "Extract 'email:password' format")
	url := flag.Bool("url", false, "Output the full matching line (contains 'term')")
	user := flag.Bool("user", false, "Extract 'username:password' format (often colon-separated)")
	tiktok := flag.Bool("tiktok", false, "Extract TikTok 'sid_guard|perf_feed_cache' cookie pairs from Netscape format files")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "A high-performance tool to search text files for specific patterns or cookie pairs.")
		fmt.Fprintln(os.Stderr, "\nOptions:")
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr, "\nExamples:")
		fmt.Fprintln(os.Stderr, "  # Search for emails in logs folder, output to emails_found.txt")
		fmt.Fprintln(os.Stderr, "  "+os.Args[0]+" -email -term \"@example.com\" -folder ./logs -output emails_found.txt")
		fmt.Fprintln(os.Stderr, "\n  # Search for URLs containing 'login' in the current directory, limit to 100 results")
		fmt.Fprintln(os.Stderr, "  "+os.Args[0]+" -url -term \"login\" -max 100")
		fmt.Fprintln(os.Stderr, "\n  # Extract TikTok cookie pairs from txt files in D:\\cookies, using 8 workers")
		fmt.Fprintln(os.Stderr, "  "+os.Args[0]+" -tiktok -folder D:\\cookies -workers 8 -show-path")
	}

	flag.Parse()

	// --- Determine Mode and Validate Inputs ---
	mode := ""
	modeCount := 0
	outputNameHint := *searchTerm // Use search term for output filename default

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
		outputNameHint = "tiktok_cookies" // Specific hint for tiktok mode
	}

	// Ensure exactly one mode is selected
	if modeCount == 0 {
		fmt.Fprintln(os.Stderr, "Error: A mode must be specified (-email, -url, -user, or -tiktok).")
		flag.Usage()
		os.Exit(1)
	}
	if modeCount > 1 {
		fmt.Fprintln(os.Stderr, "Error: Only one mode (-email, -url, -user, -tiktok) can be specified at a time.")
		os.Exit(1)
	}

	// Validate search term requirement for non-TikTok modes
	if mode != "tiktok" && *searchTerm == "" {
		fmt.Fprintf(os.Stderr, "Error: -term parameter is required for '%s' mode.\n", mode)
		flag.Usage()
		os.Exit(1)
	}
	if mode == "tiktok" && *searchTerm != "" {
		// Issue a warning, not an error, as it's just ignored
		fmt.Fprintln(os.Stderr, "Warning: -term parameter is ignored when using -tiktok mode.")
		*searchTerm = "" // Clear it internally to avoid confusion
	}

	// Validate numeric parameters
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
		// Generate filename in the current working directory
		safeHint := strings.Map(func(r rune) rune {
			// Allow letters, numbers, underscore, hyphen
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
				return r
			}
			return '_' // Replace other characters with underscore
		}, outputNameHint)
		// Prevent excessively long filenames
		if len(safeHint) > 50 {
			safeHint = safeHint[:50]
		}
		if safeHint == "" { // Handle case where term was empty or only invalid chars
			safeHint = "search_results"
		}
		timeStamp := time.Now().Format("20060102_150405")
		// Use filepath.Join with just the filename, placing it in the CWD
		*outputPath = filepath.Join(".", fmt.Sprintf("%s_%s.txt", safeHint, timeStamp))
	}

	// --- Configuration Creation ---
	maxFileSizeBytes := int64(0)
	if *maxFileSize > 0 {
		maxFileSizeBytes = *maxFileSize * 1024 * 1024 // Convert MB to Bytes
	}

	// Prepare lowercase search term variants for efficiency
	var lowerSearchTermStr string
	var lowerSearchTermBytes []byte
	if mode != "tiktok" {
		lowerSearchTermStr = strings.ToLower(*searchTerm)
		lowerSearchTermBytes = []byte(lowerSearchTermStr)
	}

	config := SearchConfig{
		FolderPath:           *folderPath,
		SearchTerm:           *searchTerm, // Store original term
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
		DebugMode:            *debugMode,
	}

	// --- Start Search ---
	fmt.Println("Starting high-performance file search...")
	fmt.Printf("  Folder:        %s\n", config.FolderPath)
	if mode != "tiktok" {
		fmt.Printf("  Search Term:   '%s' (case-insensitive)\n", config.SearchTerm)
	} else {
		fmt.Printf("  Search Target: TikTok cookie pairs ('%s', '%s') in Netscape format\n", tiktokCookieName1, tiktokCookieName2)
	}
	fmt.Printf("  Output File:   %s\n", config.OutputPath)
	fmt.Printf("  Mode:          %s\n", config.Mode)
	fmt.Printf("  Max File Size: %d MB\n", *maxFileSize) // Show user-friendly MB
	fmt.Printf("  Workers:       %d (using %d cores)\n", config.Workers, runtime.GOMAXPROCS(0))
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
		// Error already logged by searchFiles if critical (e.g., output file creation)
		// We might get non-critical errors returned here (though currently searchFiles handles them)
		fmt.Fprintf(os.Stderr, "\nError during search: %v\n", err)
		os.Exit(1)
	}

	// Final summary message
	fmt.Printf("\nSearch completed. Found %d unique matches.\n", matches)
	fmt.Printf("Results saved to: %s\n", config.OutputPath)
}

// processLine extracts the relevant information from a matched line based on the mode.
// It assumes the line contains the search term (for non-TikTok modes).
// processLine extracts the desired information based on the mode.
// It now correctly handles the user:pass format when preceded by a URL containing colons.
func processLine(line string, config SearchConfig) string {
	switch config.Mode {
	case "url":
		// Return the full line as is
		return line

	case "email":
		// Prioritize regex for common email:pass patterns
		if strings.ContainsRune(line, '@') { // Quick check
			matches := emailPassRegex.FindStringSubmatch(line)
			if len(matches) >= 3 {
				return matches[1] + ":" + matches[2] // Group 1 is email, Group 2 is pass
			}
		}

		// Check for '|' separator as an alternative common format
		if strings.ContainsRune(line, '|') && strings.ContainsRune(line, '@') {
			parts := strings.Split(line, "|")
			for i, part := range parts {
				trimmedPart := strings.TrimSpace(part)
				// Check if the part looks like an email
				if strings.ContainsRune(trimmedPart, '@') && strings.ContainsRune(trimmedPart, '.') {
					// Ensure there's a part after it to be the password
					if i+1 < len(parts) {
						// Join all subsequent parts as the password (handles '|' in password)
						password := strings.TrimSpace(strings.Join(parts[i+1:], "|"))
						if password != "" {
							return trimmedPart + ":" + password
						}
					}
					// Found email but no password after it? Break search on this line.
					break
				}
			}
		}

		// Fallback: Split by colon, find the email, take subsequent parts as password
		// This handles cases like URL:email:password or other:email:password:with:colons
		if strings.ContainsRune(line, ':') && strings.ContainsRune(line, '@') {
			parts := strings.Split(line, ":")
			if len(parts) >= 2 { // Need at least email:something
				for i, part := range parts {
					trimmedPart := strings.TrimSpace(part)
					// Check if it looks like an email
					if strings.ContainsRune(trimmedPart, '@') && strings.ContainsRune(trimmedPart, '.') {
						// Ensure there's something *after* the email part to be the password
						if i+1 < len(parts) {
							// Join all subsequent parts as the password (handles ':' in password)
							password := strings.TrimSpace(strings.Join(parts[i+1:], ":"))
							if password != "" { // Make sure password isn't empty
								return trimmedPart + ":" + password
							}
						}
						// Found email but no password after it? Break.
						break
					}
				}
			}
		}
		return "" // No email:pass format found

	case "user":
		// Handles formats like someprefix:username:password or url:username:password[:password_with_colons]
		parts := strings.Split(line, ":")
		if len(parts) >= 3 { // Need at least prefix:user:pass
			// Assume the second-to-last part is the user/credential
			user := strings.TrimSpace(parts[len(parts)-2])
			// Assume everything from the last part onwards is the password
			password := strings.TrimSpace(strings.Join(parts[len(parts)-1:], ":"))

			// Basic validation: user shouldn't be empty, password shouldn't be empty
			// Avoid extracting things like "https://domain.com:443:password" as user "443"
			// A simple heuristic: user shouldn't typically contain '/' if it's meant to be a username.
			// This isn't perfect but helps filter common URL parts.
			// Also, skip if the 'user' part contains '@' as that's likely an email.
			if user != "" && password != "" && !strings.ContainsRune(user, '/') && !strings.ContainsRune(user, '@') {
				return user + ":" + password
			}
		}
		return "" // Didn't match the expected structure

	// NOTE: "tiktok" mode is handled BEFORE calling this function in the writer goroutine.
	default:
		// Should not happen due to initial validation
		return ""
	}
}

// searchFiles sets up and manages the concurrent file search process.
func searchFiles(config SearchConfig) (int64, error) {
	// --- Output File Setup ---
	outFile, err := os.Create(config.OutputPath)
	if err != nil {
		// This is a critical error, stop immediately
		return 0, fmt.Errorf("failed to create output file '%s': %w", config.OutputPath, err)
	}
	defer outFile.Close()

	// Use a large buffer for the output file writer to minimize disk I/O calls
	writer := bufio.NewWriterSize(outFile, defaultWriterBufferSize)
	defer func() {
		// Ensure final buffer flush on exit, log potential error
		if ferr := writer.Flush(); ferr != nil {
			fmt.Fprintf(os.Stderr, "\nWarning: Error flushing output file buffer: %v\n", ferr)
		}
	}()

	// --- Concurrency Setup ---
	var wg sync.WaitGroup                           // Waits for worker goroutines to finish
	var finderWg sync.WaitGroup                     // Waits for the file finder goroutine
	var writerWg sync.WaitGroup                     // Waits for the writer goroutine
	var progressWg sync.WaitGroup                   // Waits for the progress reporting goroutine
	fileChan := make(chan string, config.Workers*2) // Channel for file paths to workers
	// resultChan carries strings formatted as "[filepath:lineNum|0] content"
	// where content is the raw line for url/email/user, or value1|value2 for tiktok
	resultChan := make(chan string, 100) // Channel for results from workers to writer
	done := make(chan struct{})          // Signal channel to stop all goroutines
	var closeDoneOnce sync.Once          // Ensures 'done' channel is closed only once

	stats := &SearchStats{StartTime: time.Now()} // Initialize statistics

	// --- Progress Reporter Goroutine ---
	progressWg.Add(1)
	go func() {
		defer progressWg.Done()
		ticker := time.NewTicker(progressUpdateInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				stats.printProgress() // Update progress periodically
			case <-done:
				stats.printProgress() // Print final stats before exiting
				fmt.Println()         // Add a newline after the final progress update
				return
			}
		}
	}()

	// --- Writer Goroutine ---
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		seen := make(map[string]struct{}) // Map for deduplicating results
		var consoleCount int              // Counter for limiting console output

		for result := range resultChan {
			// Parse the standardized result format: "[filepath:lineNum|0] content"
			fileInfo, content, parseOK := parseResultString(result)
			if !parseOK {
				if config.DebugMode { // Optionally log bad parse if debugging
					fmt.Fprintf(os.Stderr, "\nDEBUG: [Writer] Failed to parse result string: %q\n", result)
				}
				stats.ErrorsEncountered.Add(1)
				continue // Skip malformed results
			}

			var outputLine string
			isTikTokPair := (config.Mode == "tiktok")

			if isTikTokPair {
				// For TikTok mode, 'content' is already the desired "value1|value2" pair
				outputLine = content
				if outputLine == "" { // Should not happen if processFileTikTok sends valid pairs
					if config.DebugMode {
						fmt.Fprintf(os.Stderr, "\nDEBUG: [Writer] Received empty TikTok pair content from %s\n", fileInfo)
					}
					continue
				}
			} else {
				// For other modes, process the raw line content to extract desired format
				outputLine = processLine(content, config)
				if outputLine == "" {
					// Line contained the search term but didn't match the required format (e.g., email:pass)
					stats.NonMatchingSkipped.Add(1)
					continue
				}
				// Note: For URL mode, processLine returns the original 'content'
			}

			// --- Deduplication and Writing (Mutex for 'seen' map access) ---
			// Although map access isn't strictly thread-safe, we lock here mainly
			// to coordinate with potential console output and stats updates if needed later.
			// For high-contention scenarios, a sync.Map might be considered, but
			// for typical output rates, a simple mutex is often sufficient and clearer.
			// **Correction**: Map writes *are not* thread-safe. Mutex is required.
			stats.mu.Lock()

			if _, isDuplicate := seen[outputLine]; !isDuplicate {
				seen[outputLine] = struct{}{} // Mark this output line as seen

				// Construct final output line, optionally adding file path info
				var fullOutputLine string
				if config.ShowFilePath {
					// fileInfo already contains "[filepath:line/0]"
					fullOutputLine = fileInfo + " " + outputLine
				} else {
					fullOutputLine = outputLine
				}

				// Write to the buffered output file
				if _, werr := fmt.Fprintln(writer, fullOutputLine); werr != nil {
					stats.ErrorsEncountered.Add(1)
					fmt.Fprintf(os.Stderr, "\nError writing to output file: %v\n", werr)
					// Attempt to flush what we have, but continue if possible
					_ = writer.Flush()
					stats.mu.Unlock() // Unlock before continuing
					continue          // Skip console output etc. for this line
				}

				// --- Immediate flush: Reduces buffering benefits but ensures data is written sooner ---
				// Consider making this conditional (e.g., flush every N lines or T seconds)
				// For now, flushing every line for immediate feedback, similar to original code.
				if ferr := writer.Flush(); ferr != nil {
					stats.ErrorsEncountered.Add(1)
					fmt.Fprintf(os.Stderr, "\nError flushing output file: %v\n", ferr)
					// Continue processing other results despite flush error
				}
				// --- End Immediate Flush ---

				currentMatches := stats.MatchesFound.Add(1)

				// Show limited number of matches directly on the console
				if consoleCount < consoleMatchLimit {
					displayLine := outputLine
					if config.ShowFilePath {
						displayLine = fileInfo + " " + outputLine
					}
					// Add a newline before the first "Found:" to separate from progress line
					if consoleCount == 0 {
						fmt.Println() // Newline after progress bar stops
					}
					fmt.Println("Found:", displayLine) // Print the found match
					consoleCount++
					if consoleCount == consoleMatchLimit {
						fmt.Println("(Further matches will be written to file only. Press Ctrl+C to stop)")
					}
				}
				stats.mu.Unlock() // Unlock after successful processing and potential console output

				// Check if max unique occurrences reached (outside the mutex)
				if config.StopAfterMax && config.MaxOccurrences > 0 && currentMatches >= config.MaxOccurrences {
					// Signal all other goroutines to stop
					closeDoneOnce.Do(func() { close(done) })
					// No need to return here, let the loop drain remaining results if any
				}

			} else {
				// Duplicate found, increment counter
				stats.DuplicatesSkipped.Add(1)
				stats.mu.Unlock()                     // Unlock if it was a duplicate
				if config.DebugMode && isTikTokPair { // Debug duplicate TikTok pairs specifically
					// This can be noisy, use judiciously
					// fmt.Printf("\nDEBUG: [Writer] Skipping duplicate TikTok pair: %s (from %s)\n", outputLine, fileInfo)
				}
			}
		}
		// Writer goroutine finishes when resultChan is closed and loop completes
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
						return // fileChan closed, no more work
					}

					// Select appropriate processing function based on mode
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

				case <-done:
					return // Stop signal received
				}
			}
		}(i)
	}

	// --- File Finder Goroutine ---
	finderWg.Add(1)
	go func() {
		defer finderWg.Done()
		// Close fileChan when walk completes or stops, signaling workers no more files are coming
		defer close(fileChan)

		// Define allowed file extensions based on mode
		allowedExts := map[string]struct{}{ // Default extensions for most modes
			".txt": {}, ".log": {}, ".csv": {}, ".json": {}, ".xml": {}, ".html": {},
			".ini": {}, ".conf": {}, ".yaml": {}, ".yml": {}, ".cfg": {}, ".md": {},
			// Add or remove extensions as needed
		}
		// For TikTok mode, we ONLY care about .txt files (typically where Netscape cookies are stored)
		if config.Mode == "tiktok" {
			allowedExts = map[string]struct{}{".txt": {}}
			fmt.Println("  (Note: TikTok mode only searches .txt files)")
		}

		// Walk the directory tree
		err := filepath.WalkDir(config.FolderPath, func(path string, d fs.DirEntry, walkErr error) error {
			// Check if stop signal received
			select {
			case <-done:
				return fs.SkipAll // Stop walking immediately
			default:
				// Continue walking
			}

			// Handle errors during walking (e.g., permission denied)
			if walkErr != nil {
				stats.ErrorsEncountered.Add(1)
				// Log errors selectively to avoid console spam, especially for permission issues
				if !errors.Is(walkErr, os.ErrPermission) {
					fmt.Fprintf(os.Stderr, "\nWarning: Error accessing path %s: %v\n", path, walkErr)
				}
				// If it's a directory we can't enter, skip it
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				// Otherwise, continue walking other files/dirs if possible
				return nil
			}

			// Process only files (not directories)
			if !d.IsDir() {
				// Check if file extension is allowed
				ext := strings.ToLower(filepath.Ext(path))
				if _, ok := allowedExts[ext]; ok {
					stats.FilesFound.Add(1) // Increment count of potentially processable files found
					// Send the file path to the workers
					select {
					case fileChan <- path:
						// File path sent successfully
					case <-done:
						return fs.SkipAll // Stop sending if stop signal received
					}
				} else {
					// Optionally count skipped files by extension type if needed for stats
				}
			}
			return nil // Continue walking
		})

		// Check for errors during the walk itself (excluding SkipAll)
		if err != nil && !errors.Is(err, fs.SkipAll) {
			fmt.Fprintf(os.Stderr, "\nWarning: File traversal error: %v\n", err)
			stats.ErrorsEncountered.Add(1)
		}
	}()

	// --- Wait for Completion ---
	finderWg.Wait()                          // Wait for file discovery to finish (or be stopped)
	wg.Wait()                                // Wait for all worker goroutines to finish processing files
	close(resultChan)                        // Close resultChan: signals the writer goroutine to finish
	writerWg.Wait()                          // Wait for the writer goroutine to process all results and finish
	closeDoneOnce.Do(func() { close(done) }) // Ensure 'done' is closed (might have been closed by writer)
	progressWg.Wait()                        // Wait for the progress reporter to finish

	// Return the final count of unique matches found
	return stats.MatchesFound.Load(), nil
}

// parseResultString extracts file information and content from the standardized worker output string.
// Expects format: "[filepath:lineNum|0] content"
func parseResultString(result string) (fileInfo, content string, ok bool) {
	// Find the closing bracket and the space immediately after it
	idx := strings.Index(result, "] ")
	if idx == -1 || idx+2 >= len(result) {
		return "", "", false // Invalid format
	}
	// Extract the part within and including brackets as fileInfo
	fileInfo = result[:idx+1] // e.g., "[/path/to/file.txt:123]" or "[/path/to/cookies.txt:0]"
	// Extract the rest of the string as the content
	content = result[idx+2:]
	ok = true
	return
}

// processFile handles searching within a single file for email, url, or user modes.
func processFile(filePath string, config SearchConfig, resultChan chan<- string, buf []byte, stats *SearchStats, done <-chan struct{}) {
	// Recover from potential panics within this goroutine
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "\nPANIC recovered while processing file %s: %v\n", filePath, r)
			stats.ErrorsEncountered.Add(1)
			// Mark file as skipped due to panic during processing
			stats.FilesSkipped.Add(1)
		}
	}()

	// --- File Pre-checks ---
	info, err := os.Stat(filePath)
	if err != nil {
		// fmt.Fprintf(os.Stderr, "\nWarning: Cannot stat file %s: %v\n", filePath, err)
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1) // Count stat errors
		return
	}
	fileSize := info.Size()
	// Skip empty files or files exceeding the size limit
	if fileSize == 0 || (config.MaxFileSizeBytes > 0 && fileSize > config.MaxFileSizeBytes) {
		stats.FilesSkipped.Add(1)
		return
	}

	// --- File Opening ---
	file, err := os.Open(filePath)
	if err != nil {
		// fmt.Fprintf(os.Stderr, "\nWarning: Cannot open file %s: %v\n", filePath, err)
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1) // Count open errors
		return
	}
	defer file.Close() // Ensure file is closed when function returns

	// --- Scanning ---
	scanner := bufio.NewScanner(file)
	scanner.Buffer(buf, len(buf)) // Use the provided per-worker buffer

	lineNum := 0
	bytesReadInFile := int64(0)                     // Total bytes read in this specific file
	bytesReadSinceCheck := int64(0)                 // Bytes read since last 'done' check for periodic interruption
	lowerSearchBytes := config.LowerSearchTermBytes // Cache for efficiency

	for scanner.Scan() {
		lineNum++
		lineBytes := scanner.Bytes() // Get line as byte slice to avoid immediate string allocation
		lineLen := int64(len(lineBytes))
		bytesReadInFile += lineLen + 1 // +1 for the newline character (approximate)
		bytesReadSinceCheck += lineLen + 1

		// Optimization: Perform case-insensitive check using bytes.Contains on lowercase versions
		if bytes.Contains(bytes.ToLower(lineBytes), lowerSearchBytes) {
			// Potential match found, convert to string only now
			lineStr := string(lineBytes)
			// Format result consistently: [filepath:lineNum] content
			result := fmt.Sprintf("[%s:%d] %s", filePath, lineNum, lineStr)
			select {
			case resultChan <- result:
				// Result sent successfully to the writer goroutine
			case <-done:
				// Stop signal received while trying to send
				stats.BytesProcessed.Add(bytesReadInFile) // Add bytes processed in this file before stopping
				stats.FilesProcessed.Add(1)               // Mark as processed (partially)
				return                                    // Exit the function for this file
			}
		}

		// Check 'done' channel periodically (e.g., every MB) to allow interruption
		// This prevents getting stuck reading a huge file if 'stop-after-max' is hit
		if bytesReadSinceCheck > 1*1024*1024 { // Check roughly every 1MB
			select {
			case <-done:
				stats.BytesProcessed.Add(bytesReadInFile)
				stats.FilesProcessed.Add(1) // Mark as processed (partially)
				return                      // Exit loop and function
			default:
				// Continue processing
			}
			bytesReadSinceCheck = 0 // Reset counter after check
		}
	} // End of scanner loop

	// --- Post-Scanning ---
	// Add the total bytes read for this file to global stats
	stats.BytesProcessed.Add(bytesReadInFile)

	// Check for scanner errors (e.g., line too long for buffer)
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			// Specific warning for lines exceeding the buffer
			fmt.Fprintf(os.Stderr, "\nWarning: Line too long in file %s (max buffer: %d MB). File partially processed.\n", filePath, config.BufferSizeMB)
			// Mark as processed because some part might have been read
			stats.FilesProcessed.Add(1)
		} else {
			// General scanner error
			fmt.Fprintf(os.Stderr, "\nWarning: Error scanning file %s: %v\n", filePath, err)
			// Mark as skipped because processing likely failed significantly
			stats.FilesSkipped.Add(1)
		}
		stats.ErrorsEncountered.Add(1)
	} else {
		// File processed completely without scanner errors
		stats.FilesProcessed.Add(1)
	}
}

// processFileTikTok handles searching for specific TikTok cookie pairs within a single file.
// It looks for two specific cookie names associated with the tiktok.com domain in Netscape cookie format.
func processFileTikTok(filePath string, config SearchConfig, resultChan chan<- string, buf []byte, stats *SearchStats, done <-chan struct{}) {
	// Recover from potential panics within this goroutine
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "\nPANIC recovered while processing file %s (TikTok mode): %v\n", filePath, r)
			stats.ErrorsEncountered.Add(1)
			stats.FilesSkipped.Add(1) // Mark as skipped due to panic
		}
	}()

	// --- File Pre-checks ---
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

	// --- File Opening ---
	file, err := os.Open(filePath)
	if err != nil {
		stats.FilesSkipped.Add(1)
		stats.ErrorsEncountered.Add(1)
		return
	}
	defer file.Close()

	// --- Scanning ---
	scanner := bufio.NewScanner(file)
	// Use the main per-worker buffer passed in
	scanner.Buffer(buf, len(buf))

	var foundVal1, foundVal2 string // Store the values of the target cookies when found
	lineNumber := 0                 // Track line number for debugging
	bytesReadInFile := int64(0)     // Total bytes read in this file
	bytesReadSinceCheck := int64(0) // Bytes read since last 'done' check
	foundPair := false              // Flag to indicate if both cookies have been found

	for scanner.Scan() {
		// If we already found the pair in this file, no need to scan further
		if foundPair {
			break
		}

		lineNumber++
		line := scanner.Text() // Get line as string for splitting and checks
		lineLen := int64(len(line))
		bytesReadInFile += lineLen + 1
		bytesReadSinceCheck += lineLen + 1

		// --- TikTok Specific Logic ---
		// Quick checks: skip comments, empty lines, or lines unlikely to be Netscape format
		if line == "" || line[0] == '#' || !strings.Contains(line, "\t") {
			continue
		}

		// Split the line by tab (Netscape format delimiter)
		fields := strings.Split(line, "\t")

		// Check minimum field count and if the domain matches tiktokTargetDomain
		// Index 0 is the domain field in Netscape format
		if len(fields) < tiktokMinFields || !strings.Contains(fields[0], tiktokTargetDomain) {
			continue // Skip lines not matching basic criteria
		}

		// Ensure the indices for cookie name and value are valid before accessing
		if len(fields) <= tiktokCookieNameIndex1 || len(fields) <= tiktokCookieValueIndex1 {
			if config.DebugMode {
				fmt.Fprintf(os.Stderr, "\nDEBUG: Short line (%d fields) in %s:%d: %q\n", len(fields), filePath, lineNumber, line)
			}
			continue // Skip malformed lines where indices would be out of bounds
		}

		cookieName := fields[tiktokCookieNameIndex1]
		cookieValue := fields[tiktokCookieValueIndex1]

		// Check if this line contains one of the target cookies (if not already found)
		if foundVal1 == "" && cookieName == tiktokCookieName1 {
			foundVal1 = cookieValue
			if config.DebugMode {
				fmt.Printf("\nDEBUG: Found '%s' in '%s' (Line %d)\n", tiktokCookieName1, filePath, lineNumber)
			}
		}
		if foundVal2 == "" && cookieName == tiktokCookieName2 {
			foundVal2 = cookieValue
			if config.DebugMode {
				fmt.Printf("\nDEBUG: Found '%s' in '%s' (Line %d)\n", tiktokCookieName2, filePath, lineNumber)
			}
		}
		// --- End TikTok Logic ---

		// Check if BOTH cookies have now been found in this file
		if foundVal1 != "" && foundVal2 != "" {
			foundPair = true // Set flag to stop scanning loop early
			if config.DebugMode {
				fmt.Printf("\nDEBUG: Found BOTH '%s' and '%s' in '%s'. Sending result.\n", tiktokCookieName1, tiktokCookieName2, filePath)
			}
			// Format the result: value1|value2
			pairResult := fmt.Sprintf("%s|%s", foundVal1, foundVal2)
			// Standardize output format: [filepath:0] content (use 0 for lineNum as it's file-level)
			resultString := fmt.Sprintf("[%s:0] %s", filePath, pairResult)

			select {
			case resultChan <- resultString:
				// Sent the pair successfully
				// We can break the loop now as we found what we needed from this file
			case <-done:
				// Stop signal received before sending
				// Don't send, just exit
			}
			// Break regardless of whether sent or stopped, as we found the pair
			break
		}

		// Periodic check for done signal (every ~1MB)
		if bytesReadSinceCheck > 1*1024*1024 {
			select {
			case <-done:
				// Stop signal received during scan
				stats.BytesProcessed.Add(bytesReadInFile)
				stats.FilesProcessed.Add(1) // Mark as processed (partially)
				return                      // Exit function
			default:
				// Continue processing
			}
			bytesReadSinceCheck = 0 // Reset counter
		}
	} // End of scanner loop

	// --- Post-Scanning ---
	// Add total bytes read for this file
	stats.BytesProcessed.Add(bytesReadInFile)

	// Check for scanner errors
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			fmt.Fprintf(os.Stderr, "\nWarning: Line too long in file %s (TikTok mode, max buffer: %d MB).\n", filePath, config.BufferSizeMB)
			// Mark as processed because we might have found the pair before hitting the long line
			stats.FilesProcessed.Add(1)
		} else {
			fmt.Fprintf(os.Stderr, "\nWarning: Error scanning file %s (TikTok mode): %v\n", filePath, err)
			stats.FilesSkipped.Add(1) // Mark as skipped due to significant scan error
		}
		stats.ErrorsEncountered.Add(1)
	} else {
		// Reached end of file without scanner error
		// Mark as processed, regardless of whether the pair was found or not
		stats.FilesProcessed.Add(1)
		if !foundPair && config.DebugMode {
			// Optionally log if a file was fully scanned but the pair wasn't found
			// fmt.Printf("\nDEBUG: File %s fully scanned, TikTok pair not found.\n", filePath)
		}
	}
}
