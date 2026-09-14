with open('README.md', 'r') as f:
    content = f.read()

content = content.replace('##  ', '## ')
content = content.replace('%  ', '% ')

with open('README.md', 'w') as f:
    f.write(content)
