def fix_line_endings(filename):
    """Convert CRLF to LF in the given file."""
    with open(filename, 'rb') as file:
        content = file.read()
    
    # Replace CRLF with LF
    content = content.replace(b'\r\n', b'\n')
    
    with open(filename, 'wb') as file:
        file.write(content)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fix_line_endings.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    fix_line_endings(filename)
    print(f"Converted line endings in {filename} to Unix format") 