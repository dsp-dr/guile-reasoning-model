#!/usr/bin/env bash
# decompose-book.sh - Complete book decomposition pipeline for agent processing

set -euo pipefail

SOURCE_PDF="tmp/Build_a_Reasoning_Model_(From_Scratch)_v1_MEAP.pdf"
DECOMP_DIR="decomp"
TOTAL_PAGES=94

echo "=== Book Decomposition Pipeline ==="
echo "Source: $SOURCE_PDF"
echo "Output: $DECOMP_DIR/"
echo "Total pages: $TOTAL_PAGES"
echo

# Verify source exists
if [[ ! -f "$SOURCE_PDF" ]]; then
    echo "Error: Source PDF not found at $SOURCE_PDF"
    exit 1
fi

# Setup directory structure
echo "Creating directory structure..."
mkdir -p "$DECOMP_DIR"/{chapters,sections,focus-areas,images/{pages,thumbs,diagrams,code},metadata}

# Chapter-level splits (if pdf-splitter supports it)
echo "Extracting chapter-level PDFs..."
if [[ -x "./pdf-splitter.py" ]]; then
    ./pdf-splitter.py split "$SOURCE_PDF" --by-chapter --output-dir "$DECOMP_DIR/chapters/" || echo "Chapter split failed, continuing..."
else
    echo "pdf-splitter.py not found, skipping automated chapter extraction"
fi

# Section-level splits (5-page chunks)
echo "Creating section-level chunks..."
for ((start=1; start<=TOTAL_PAGES; start+=5)); do
    end=$((start + 4))
    if [[ $end -gt $TOTAL_PAGES ]]; then
        end=$TOTAL_PAGES
    fi
    
    section_name=$(printf "section-%02d-pages-%03d-%03d" $((start/5 + 1)) "$start" "$end")
    
    # Extract pages using pdftk if available, otherwise use python
    if command -v pdftk >/dev/null 2>&1; then
        pdftk "$SOURCE_PDF" cat "$start-$end" output "$DECOMP_DIR/sections/${section_name}.pdf" 2>/dev/null || {
            echo "pdftk failed for section $section_name, trying python..."
            python3 -c "
import PyPDF2
reader = PyPDF2.PdfReader('$SOURCE_PDF')
writer = PyPDF2.PdfWriter()
for i in range($start-1, min($end, len(reader.pages))):
    writer.add_page(reader.pages[i])
with open('$DECOMP_DIR/sections/${section_name}.pdf', 'wb') as f:
    writer.write(f)
print('Created $DECOMP_DIR/sections/${section_name}.pdf')
"
        }
    else
        python3 -c "
import PyPDF2
reader = PyPDF2.PdfReader('$SOURCE_PDF')
writer = PyPDF2.PdfWriter()
for i in range($start-1, min($end, len(reader.pages))):
    writer.add_page(reader.pages[i])
with open('$DECOMP_DIR/sections/${section_name}.pdf', 'wb') as f:
    writer.write(f)
print('Created $DECOMP_DIR/sections/${section_name}.pdf')
"
    fi
done

# Focus area extracts (based on likely content distribution)
echo "Creating focus-area extracts..."

# Tokenization (estimated pages 20-35)
python3 -c "
import PyPDF2
reader = PyPDF2.PdfReader('$SOURCE_PDF')
writer = PyPDF2.PdfWriter()
for i in range(19, min(35, len(reader.pages))):  # 0-indexed
    writer.add_page(reader.pages[i])
with open('$DECOMP_DIR/focus-areas/tokenization.pdf', 'wb') as f:
    writer.write(f)
print('Created tokenization focus area')
"

# Generation (estimated pages 35-55)  
python3 -c "
import PyPDF2
reader = PyPDF2.PdfReader('$SOURCE_PDF')
writer = PyPDF2.PdfWriter()
for i in range(34, min(55, len(reader.pages))):
    writer.add_page(reader.pages[i])
with open('$DECOMP_DIR/focus-areas/generation.pdf', 'wb') as f:
    writer.write(f)
print('Created generation focus area')
"

# KV Caching (estimated pages 55-70)
python3 -c "
import PyPDF2
reader = PyPDF2.PdfReader('$SOURCE_PDF')
writer = PyPDF2.PdfWriter()
for i in range(54, min(70, len(reader.pages))):
    writer.add_page(reader.pages[i])
with open('$DECOMP_DIR/focus-areas/kv-caching.pdf', 'wb') as f:
    writer.write(f)
print('Created KV caching focus area')
"

# Convert to images for visual analysis
echo "Converting pages to images..."
if command -v magick >/dev/null 2>&1; then
    for ((i=1; i<=TOTAL_PAGES; i++)); do
        page_num=$(printf "%03d" $i)
        echo -n "."
        
        # Full resolution page
        magick "$SOURCE_PDF[$((i-1))]" -density 300 -quality 95 \
            "$DECOMP_DIR/images/pages/page-${page_num}.png" 2>/dev/null || {
            echo "Failed to convert page $i"
        }
        
        # Thumbnail
        if [[ -f "$DECOMP_DIR/images/pages/page-${page_num}.png" ]]; then
            magick "$DECOMP_DIR/images/pages/page-${page_num}.png" \
                -thumbnail 200x260 \
                "$DECOMP_DIR/images/thumbs/page-${page_num}-thumb.png" 2>/dev/null
        fi
        
        # Progress indicator
        if [[ $((i % 10)) -eq 0 ]]; then
            echo " $i/$TOTAL_PAGES"
        fi
    done
    echo
else
    echo "ImageMagick 'magick' command not found, skipping image conversion"
fi

# Create metadata
echo "Generating metadata..."
cat > "$DECOMP_DIR/metadata/decomposition-summary.txt" << EOF
Book Decomposition Summary
==========================
Generated: $(date)
Source: $SOURCE_PDF
Total Pages: $TOTAL_PAGES

Directory Structure:
- chapters/: Chapter-level PDFs
- sections/: 5-page section chunks ($(ls "$DECOMP_DIR/sections/" 2>/dev/null | wc -l) files)
- focus-areas/: Implementation-focused extracts (3 files)
- images/pages/: Full-resolution page images ($(ls "$DECOMP_DIR/images/pages/" 2>/dev/null | wc -l) files)  
- images/thumbs/: Thumbnail images ($(ls "$DECOMP_DIR/images/thumbs/" 2>/dev/null | wc -l) files)

Focus Areas:
- tokenization.pdf: Pages 20-35 (tokenization implementation)
- generation.pdf: Pages 35-55 (text generation)
- kv-caching.pdf: Pages 55-70 (KV cache optimization)

Usage:
- Individual agents can process sections/ files independently
- Focus areas provide concentrated material for specific implementations
- Images enable visual analysis of diagrams and code examples
- Thumbnails provide quick overview for task planning

Next Steps:
1. Review decomposed sections for quality
2. Assign sections to specialized agents
3. Begin parallel implementation
4. Coordinate integration between components
EOF

# Create agent task index
echo "Creating agent task index..."
cat > "$DECOMP_DIR/metadata/agent-tasks.org" << 'EOF'
#+TITLE: Agent Task Index
#+DATE: $(date)

* Chapter-Level Tasks

| Agent | Input | Output | Status |
|-------|-------|--------|--------|
| ConceptAgent | chapters/chapter-01.pdf | src/theory/ | Pending |
| CoreAgent | chapters/chapter-02.pdf | src/tokenizer/, src/model/, src/generation/ | Pending |
| EvalAgent | TBD (Chapter 3) | src/evaluation/ | Waiting for MEAP |
| InferenceAgent | TBD (Chapter 4) | src/inference/ | Anticipated |

* Section-Level Tasks  

| Section | Pages | Focus | Assigned Agent | Status |
|---------|-------|-------|---------------|--------|
| 01 | 1-5 | Introduction, reasoning concepts | ConceptAgent | Pending |
| 02 | 6-10 | Training pipeline overview | ConceptAgent | Pending |
| 03 | 11-15 | Pattern matching theory | ConceptAgent | Pending |
| 04 | 16-20 | LLM generation intro | CoreAgent | Pending |
| 05 | 21-25 | Environment setup | CoreAgent | Pending |
| 06 | 26-30 | Input preparation/tokenization | TokenAgent | Pending |
| 07 | 31-35 | Model loading | ModelAgent | Pending |
| 08 | 36-40 | Sequential generation | GenAgent | Pending |
| 09 | 41-45 | Generation function coding | GenAgent | Pending |
| 10 | 46-50 | KV caching theory | CacheAgent | Pending |
| 11 | 51-55 | KV caching implementation | CacheAgent | Pending |
| 12 | 56-60 | PyTorch compilation (N/A for Guile) | -- | Skip |
| ... | ... | ... | ... | ... |

* Focus-Area Tasks

| Area | Input | Specialist | Deliverables |
|------|-------|------------|--------------|
| Tokenization | focus-areas/tokenization.pdf | TokenAgent | Complete tokenizer module |
| Generation | focus-areas/generation.pdf | GenAgent | Generation system |
| KV Caching | focus-areas/kv-caching.pdf | CacheAgent | Cache implementation |

EOF

echo
echo "=== Decomposition Complete ==="
echo "Results in: $DECOMP_DIR/"
echo "Summary: $DECOMP_DIR/metadata/decomposition-summary.txt"
echo "Agent tasks: $DECOMP_DIR/metadata/agent-tasks.org"
echo
echo "Ready for parallel agent processing!"