"""Convert paper.md to paper.html with KaTeX math support."""
import markdown
import re

def convert():
    with open('paper.md', 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # Protect math blocks from markdown processing
    math_blocks = []
    
    def save_display_math(m):
        math_blocks.append(m.group(1))
        return f'\x00DISPLAYMATH{len(math_blocks)-1}\x00'
    
    def save_inline_math(m):
        math_blocks.append(m.group(1))
        return f'\x00INLINEMATH{len(math_blocks)-1}\x00'
    
    # Save display math ($$...$$) first, then inline ($...$)
    md_text = re.sub(r'\$\$(.*?)\$\$', save_display_math, md_text, flags=re.DOTALL)
    md_text = re.sub(r'\$([^\$\n]+?)\$', save_inline_math, md_text)
    
    # Convert markdown to HTML
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    # Restore math blocks
    for i, block in enumerate(math_blocks):
        html_body = html_body.replace(
            f'\x00DISPLAYMATH{i}\x00',
            f'<span class="katex-display"><span class="katex">{block}</span></span>'
        )
        html_body = html_body.replace(
            f'\x00INLINEMATH{i}\x00',
            f'<span class="katex">{block}</span>'
        )
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>基于贝叶斯推断的12口弹珠机自适应投注策略研究</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px 40px;
            line-height: 1.8;
            color: #333;
            background: #fff;
        }}
        h1 {{ text-align: center; font-size: 1.8em; margin-bottom: 0.3em; }}
        h2 {{ border-bottom: 2px solid #2196F3; padding-bottom: 0.3em; margin-top: 2em; color: #1565C0; }}
        h3 {{ color: #1976D2; margin-top: 1.5em; }}
        h4 {{ color: #1E88E5; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: center;
        }}
        th {{ background: #E3F2FD; font-weight: 600; }}
        tr:nth-child(even) {{ background: #FAFAFA; }}
        blockquote {{
            border-left: 4px solid #2196F3;
            margin: 1em 0;
            padding: 0.5em 1em;
            background: #F5F5F5;
            color: #555;
        }}
        code {{
            background: #F5F5F5;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        pre {{
            background: #263238;
            color: #EEFFFF;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        pre code {{ background: none; color: inherit; padding: 0; }}
        strong {{ color: #D32F2F; }}
        hr {{ border: none; border-top: 1px solid #E0E0E0; margin: 2em 0; }}
        .katex-display {{ display: block; text-align: center; margin: 1em 0; }}
        @media print {{
            body {{ max-width: none; padding: 10px; }}
            h2 {{ page-break-before: always; }}
            h2:first-of-type {{ page-break-before: avoid; }}
        }}
    </style>
</head>
<body>
{html_body}
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script>
        document.querySelectorAll('.katex').forEach(el => {{
            try {{
                katex.render(el.textContent, el, {{
                    throwOnError: false,
                    displayMode: el.parentElement.classList.contains('katex-display')
                }});
            }} catch(e) {{}}
        }});
    </script>
</body>
</html>"""
    
    with open('paper.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("paper.html generated successfully!")

if __name__ == '__main__':
    convert()
