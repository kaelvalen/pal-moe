import sys
import re

def remove_emojis(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove specific emojis used
    emojis = ['🚀', '📊', '🛠', '⚙️', '📜', '⚖️', '🔥', '🏆']
    for emoji in emojis:
        content = content.replace(emoji, '')
        
    # Remove any stray spaces left behind by emoji removal at the end of lines
    content = re.sub(r' \n', '\n', content)
    
    with open(filepath, 'w') as f:
        f.write(content)

remove_emojis('README.md')
