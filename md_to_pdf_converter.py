#!/usr/bin/env python3
"""
Markdown to PDF Converter Script
Supports multiple conversion methods with professional formatting
"""

import os
import sys
import argparse
from pathlib import Path

# Method 1: Using markdown + weasyprint (Recommended for best formatting)
def convert_with_weasyprint(md_file, output_file=None):
    """Convert markdown to PDF using markdown + weasyprint (best quality)"""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("pip install markdown weasyprint")
        return False
    
    if output_file is None:
        output_file = md_file.replace('.md', '.pdf')
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML with extensions
    html = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc',
            'markdown.extensions.extra'
        ]
    )
    
    # Professional CSS styling
    css = CSS(string="""
    @page {
        size: A4;
        margin: 1in;
        @bottom-center {
            content: counter(page);
            font-size: 9pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Arial', 'Helvetica', sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: none;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        page-break-after: avoid;
    }
    
    h1 {
        font-size: 24pt;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3em;
    }
    
    h2 {
        font-size: 20pt;
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 0.2em;
    }
    
    h3 {
        font-size: 16pt;
        color: #34495e;
    }
    
    h4 {
        font-size: 14pt;
        color: #7f8c8d;
    }
    
    code {
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 10pt;
    }
    
    pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1em;
        overflow-x: auto;
        margin: 1em 0;
        page-break-inside: avoid;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        page-break-inside: avoid;
    }
    
    table th, table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    
    table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        margin: 1em 0;
        padding-left: 1em;
        color: #7f8c8d;
        font-style: italic;
    }
    
    ul, ol {
        margin: 1em 0;
        padding-left: 2em;
    }
    
    li {
        margin: 0.5em 0;
    }
    
    strong {
        color: #2c3e50;
    }
    
    em {
        color: #7f8c8d;
    }
    
    .page-break {
        page-break-before: always;
    }
    
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1em auto;
    }
    """)
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Graph Traversal Documentation</title>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Convert to PDF
    font_config = FontConfiguration()
    HTML(string=full_html).write_pdf(
        output_file,
        stylesheets=[css],
        font_config=font_config
    )
    
    print(f"✅ PDF created successfully: {output_file}")
    return True

# Method 2: Using markdown2pdf (Simple alternative)
def convert_with_markdown2pdf(md_file, output_file=None):
    """Convert using markdown2pdf library (simpler but less control)"""
    try:
        from markdown2pdf import convert
    except ImportError:
        print("❌ Missing dependency. Install with:")
        print("pip install markdown2pdf")
        return False
    
    if output_file is None:
        output_file = md_file.replace('.md', '.pdf')
    
    convert(md_file, output_file)
    print(f"✅ PDF created successfully: {output_file}")
    return True

# Method 3: Using pandoc (requires pandoc installation)
def convert_with_pandoc(md_file, output_file=None):
    """Convert using pandoc (requires separate pandoc installation)"""
    import subprocess
    
    if output_file is None:
        output_file = md_file.replace('.md', '.pdf')
    
    try:
        # Check if pandoc is installed
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        
        # Convert with pandoc
        cmd = [
            'pandoc',
            md_file,
            '-o', output_file,
            '--pdf-engine=xelatex',  # or 'pdflatex'
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '--highlight-style=github',
            '--table-of-contents',
            '--number-sections'
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✅ PDF created successfully: {output_file}")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Pandoc not found. Install from: https://pandoc.org/installing.html")
        return False
    except Exception as e:
        print(f"❌ Error with pandoc: {e}")
        return False

# Method 4: Using pdfkit + markdown (requires wkhtmltopdf)
def convert_with_pdfkit(md_file, output_file=None):
    """Convert using pdfkit (requires wkhtmltopdf installation)"""
    try:
        import markdown
        import pdfkit
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("pip install markdown pdfkit")
        print("Also install wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
        return False
    
    if output_file is None:
        output_file = md_file.replace('.md', '.pdf')
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    
    # PDF options
    options = {
        'page-size': 'A4',
        'margin-top': '1in',
        'margin-right': '1in',
        'margin-bottom': '1in',
        'margin-left': '1in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    try:
        pdfkit.from_string(html, output_file, options=options)
        print(f"✅ PDF created successfully: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error with pdfkit: {e}")
        print("Make sure wkhtmltopdf is installed and in PATH")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF')
    parser.add_argument('input_file', help='Input markdown file (.md)')
    parser.add_argument('-o', '--output', help='Output PDF file (optional)')
    parser.add_argument('-m', '--method', 
                       choices=['weasyprint', 'markdown2pdf', 'pandoc', 'pdfkit'],
                       default='weasyprint',
                       help='Conversion method (default: weasyprint)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    # Convert based on method
    methods = {
        'weasyprint': convert_with_weasyprint,
        'markdown2pdf': convert_with_markdown2pdf,
        'pandoc': convert_with_pandoc,
        'pdfkit': convert_with_pdfkit
    }
    
    success = methods[args.method](args.input_file, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("📄 Markdown to PDF Converter")
        print("=" * 40)
        print()
        print("Usage examples:")
        print("python md_to_pdf.py document.md")
        print("python md_to_pdf.py document.md -o output.pdf")
        print("python md_to_pdf.py document.md -m pandoc")
        print()
        print("Available methods:")
        print("• weasyprint (recommended) - Best formatting")
        print("• pandoc - Requires pandoc installation") 
        print("• pdfkit - Requires wkhtmltopdf installation")
        print("• markdown2pdf - Simple alternative")
        print()
        print("Installation commands:")
        print("pip install markdown weasyprint  # For weasyprint method")
        print("pip install markdown2pdf         # For markdown2pdf method")
        print("pip install markdown pdfkit      # For pdfkit method")
        print()
        
        # Interactive mode
        md_file = input("Enter markdown file path: ").strip()
        if md_file and os.path.exists(md_file):
            convert_with_weasyprint(md_file)
        else:
            print("❌ File not found")
        
        sys.exit(0)
    
    sys.exit(main())
