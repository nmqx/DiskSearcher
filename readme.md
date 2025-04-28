# DiskSearcher

![Go](https://img.shields.io/badge/language-Go-blue?logo=go) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

A high-performance, concurrent CLI tool to **search** directories for:

- **Emails** (`email:password` pairs)
- **URLs** (full lines containing your search term)
- **User** credentials (`username:password`)
- **TikTok cookie pairs** (`sid_guard|perf_feed_cache` from Netscape format files)

Built in Go for speed and scalability, DiskSearcher uses multiple CPU cores, buffered I/O, and optional early stopping to handle directories containing thousands of large files with ease.

---

## 🚀 Features

- **Multi-Mode Search**: `email`, `url`, `user`, or `tiktok` cookie extraction.
- **Concurrent Workers**: Leverage all CPU cores with customizable goroutine count.
- **Buffered I/O**: Tunable buffer sizes for efficient large-file scanning.
- **Progress Reporting**: Live stats on files found, processed, errors, and match rate.
- **Deduplication**: Unique-match filtering (console and output file).
- **Custom Output**: Results saved to a file (auto-named or your choice).
- **Early Stop**: Stop after a maximum number of unique hits.
- **Debug Logging**: In-depth logging for TikTok pairing logic.
- **Flexible File Size Limits**: Skip files over a certain size to avoid resource hogging.

---

## ⚙️ Installation

### From source

```bash
# Clone the repository
git clone https://github.com/nmqx/DiskSearcher.git
cd DiskSearcher

# Build the binary
go build -o disksearcher DiskSearcher.go
```

### Using `go install`

```bash
go install github.com/nmqx/DiskSearcher@latest
```

Add `$GOPATH/bin` (or your module-bin) to your `PATH` to run `disksearcher` anywhere.

---

## 💻 Usage

```bash
disksearcher [options]
```

### Modes (choose exactly one)

- `-email`     Extract `email:password` pairs
- `-url`       Print full lines containing the search term
- `-user`      Extract `username:password` pairs
- `-tiktok`    Extract `sid_guard|perf_feed_cache` cookie pairs (ignores `-term`)

### Common Options

| Flag               | Description                                                           | Default        |
| ------------------ | --------------------------------------------------------------------- | -------------- |
| `-folder`          | Input folder to search                                                | `.` (cwd)      |
| `-term`            | Term to search for (required for non-TikTok modes, case-insensitive)  | _none_         |
| `-output`          | Output file path                                                       | Auto-generated |
| `-workers`         | Number of concurrent workers (goroutines)                              | CPU cores      |
| `-buffer`          | Buffer size per worker (MB)                                            | 32             |
| `-max-file-size`   | Max file size to process (MB, 0 = unlimited)                           | 15000          |
| `-max`             | Stop after this many unique matches (0 = unlimited)                    | 0              |
| `-stop-after-max`  | Stop searching all files once `-max` unique matches are found         | false          |
| `-show-path`       | Include `[filepath:line]` prefix in output                             | false          |
| `-debug`           | Enable detailed debug output (TikTok mode)                             | false          |
| `-v`               | Enable verbose logging (placeholder)                                   | false          |

### Examples

1. **Search for emails** containing `@example.com` in `./logs`:

   ```bash
   disksearcher -email -term "@example.com" -folder ./logs -output emails_found.txt
   ```

2. **Find URLs** with `login`, stop after 100 results:

   ```bash
   disksearcher -url -term "login" -max 100 -stop-after-max
   ```

3. **Extract TikTok cookie pairs** from `~/cookies` using 8 workers and include file paths:

   ```bash
   disksearcher -tiktok -folder ~/cookies -workers 8 -show-path -debug
   ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

- Fork the repository
- Create a new branch for your feature or bugfix
- Submit a pull request

Please follow standard GitHub flow and keep changes focused.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## ✉️ Author

**nmqx**

Feel free to open an issue or reach out via GitHub for questions and support.

