# Python Regular Expressions

This document provides a comprehensive guide to Python regular expressions (regex) using the `re` module with syntax and usage examples.

## Basic Regex Operations

### Importing and Basic Matching

```python
import re

# match() - match at beginning of string
result = re.match(r'Hello', 'Hello, World!')
print(result)                               # <re.Match object>
print(result.group())                       # Hello

result = re.match(r'World', 'Hello, World!')
print(result)                               # None (doesn't match at start)

# search() - search anywhere in string
result = re.search(r'World', 'Hello, World!')
print(result)                               # <re.Match object>
print(result.group())                       # World

# findall() - find all matches
text = "The rain in Spain falls mainly in the plain"
matches = re.findall(r'ain', text)
print(matches)                              # ['ain', 'ain', 'ain', 'ain']

# finditer() - iterator of match objects
for match in re.finditer(r'ain', text):
    print(f"Found at {match.start()}: {match.group()}")

# fullmatch() - match entire string
result = re.fullmatch(r'Hello, World!', 'Hello, World!')
print(result)                               # <re.Match object>

result = re.fullmatch(r'Hello', 'Hello, World!')
print(result)                               # None (doesn't match full string)
```

### String Substitution

```python
# sub() - replace matches
text = "The rain in Spain"
result = re.sub(r'ain', 'XXX', text)
print(result)                               # The rXXX in SpXXX

# Limit number of replacements
result = re.sub(r'ain', 'XXX', text, count=1)
print(result)                               # The rXXX in Spain

# subn() - returns tuple (new_string, number_of_replacements)
result = re.subn(r'ain', 'XXX', text)
print(result)                               # ('The rXXX in SpXXX', 2)

# Using function for replacement
def replace_func(match):
    return match.group().upper()

result = re.sub(r'ain', replace_func, text)
print(result)                               # The rAIN in SpAIN
```

### Splitting Strings

```python
# split() - split by pattern
text = "Hello,World;Python:Regex"
result = re.split(r'[,;:]', text)
print(result)                               # ['Hello', 'World', 'Python', 'Regex']

# Limit number of splits
result = re.split(r'[,;:]', text, maxsplit=2)
print(result)                               # ['Hello', 'World', 'Python:Regex']

# Keep separators
result = re.split(r'([,;:])', text)
print(result)                               # ['Hello', ',', 'World', ';', 'Python', ':', 'Regex']

# Split on whitespace
text = "Hello   World\tPython\nRegex"
result = re.split(r'\s+', text)
print(result)                               # ['Hello', 'World', 'Python', 'Regex']
```

## Pattern Syntax

### Literal Characters

```python
# Literal matching
pattern = r'hello'
print(re.search(pattern, 'hello world'))    # <re.Match object>

# Case sensitive by default
print(re.search(r'Hello', 'hello'))         # None

# Special characters need escaping
pattern = r'\$100'                          # Match $100
print(re.search(pattern, 'Price: $100'))    # <re.Match object>

# Escape special characters: . ^ $ * + ? { } [ ] \ | ( )
pattern = r'\(hello\)'                      # Match (hello)
print(re.search(pattern, '(hello)'))        # <re.Match object>
```

### Character Classes

```python
# [abc] - match any character in set
pattern = r'[aeiou]'
print(re.findall(pattern, 'hello'))         # ['e', 'o']

# [^abc] - match any character NOT in set
pattern = r'[^aeiou]'
print(re.findall(pattern, 'hello'))         # ['h', 'l', 'l']

# [a-z] - match range
pattern = r'[a-z]+'
print(re.findall(pattern, 'Hello World'))   # ['ello', 'orld']

# [0-9] - match digits
pattern = r'[0-9]+'
print(re.findall(pattern, 'Room 123'))      # ['123']

# [a-zA-Z] - match any letter
pattern = r'[a-zA-Z]+'
print(re.findall(pattern, 'Hello123World')) # ['Hello', 'World']

# Multiple ranges
pattern = r'[a-zA-Z0-9]+'
print(re.findall(pattern, 'Hello123World')) # ['Hello123World']
```

### Predefined Character Classes

```python
# \d - digit [0-9]
pattern = r'\d+'
print(re.findall(pattern, 'Room 123'))      # ['123']

# \D - non-digit [^0-9]
pattern = r'\D+'
print(re.findall(pattern, 'Room 123'))      # ['Room ', ' ']

# \w - word character [a-zA-Z0-9_]
pattern = r'\w+'
print(re.findall(pattern, 'hello_world123'))  # ['hello_world123']

# \W - non-word character
pattern = r'\W+'
print(re.findall(pattern, 'hello, world!')) # [', ', '!']

# \s - whitespace [ \t\n\r\f\v]
pattern = r'\s+'
print(re.findall(pattern, 'hello world\tpython'))  # [' ', '\t']

# \S - non-whitespace
pattern = r'\S+'
print(re.findall(pattern, 'hello world'))   # ['hello', 'world']

# . - any character except newline
pattern = r'h.llo'
print(re.search(pattern, 'hello'))          # <re.Match object>
print(re.search(pattern, 'hallo'))          # <re.Match object>
print(re.search(pattern, 'h\nllo'))         # None
```

### Anchors and Boundaries

```python
# ^ - start of string
pattern = r'^Hello'
print(re.search(pattern, 'Hello World'))    # <re.Match object>
print(re.search(pattern, 'Say Hello'))      # None

# $ - end of string
pattern = r'World$'
print(re.search(pattern, 'Hello World'))    # <re.Match object>
print(re.search(pattern, 'World Hello'))    # None

# \b - word boundary
pattern = r'\bhello\b'
print(re.search(pattern, 'hello world'))    # <re.Match object>
print(re.search(pattern, 'helloworld'))     # None

# \B - not word boundary
pattern = r'\Bhello'
print(re.search(pattern, 'say hello'))      # None
print(re.search(pattern, 'sayhello'))       # <re.Match object>

# \A - start of string (like ^)
pattern = r'\AHello'
print(re.search(pattern, 'Hello\nWorld', re.MULTILINE))  # <re.Match object>

# \Z - end of string (like $)
pattern = r'World\Z'
print(re.search(pattern, 'Hello\nWorld', re.MULTILINE))  # <re.Match object>
```

### Quantifiers

```python
# * - zero or more
pattern = r'ab*c'
print(re.findall(pattern, 'ac abc abbc'))   # ['ac', 'abc', 'abbc']

# + - one or more
pattern = r'ab+c'
print(re.findall(pattern, 'ac abc abbc'))   # ['abc', 'abbc']

# ? - zero or one
pattern = r'ab?c'
print(re.findall(pattern, 'ac abc abbc'))   # ['ac', 'abc']

# {n} - exactly n times
pattern = r'ab{2}c'
print(re.findall(pattern, 'ac abc abbc abbbc'))  # ['abbc']

# {n,} - n or more times
pattern = r'ab{2,}c'
print(re.findall(pattern, 'ac abc abbc abbbc'))  # ['abbc', 'abbbc']

# {n,m} - between n and m times
pattern = r'ab{1,3}c'
print(re.findall(pattern, 'ac abc abbc abbbc'))  # ['abc', 'abbc', 'abbbc']

# Greedy vs non-greedy
text = '<div>content</div>'
print(re.findall(r'<.*>', text))            # ['<div>content</div>'] (greedy)
print(re.findall(r'<.*?>', text))           # ['<div>', '</div>'] (non-greedy)

# Make all quantifiers non-greedy
print(re.findall(r'<.+?>', text))           # ['<div>', '</div>']
print(re.findall(r'<.+>', text))            # ['<div>content</div>']
```

### Groups and Capturing

```python
# () - capturing group
pattern = r'(\d+)-(\d+)-(\d+)'
match = re.search(pattern, 'Date: 2024-01-15')
print(match.group(0))                       # 2024-01-15 (entire match)
print(match.group(1))                       # 2024
print(match.group(2))                       # 01
print(match.group(3))                       # 15
print(match.groups())                       # ('2024', '01', '15')

# Named groups (?P<name>...)
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
match = re.search(pattern, 'Date: 2024-01-15')
print(match.group('year'))                  # 2024
print(match.group('month'))                 # 01
print(match.group('day'))                   # 15
print(match.groupdict())                    # {'year': '2024', 'month': '01', 'day': '15'}

# Non-capturing group (?:...)
pattern = r'(?:Mr|Mrs|Ms)\. (\w+)'
match = re.search(pattern, 'Hello, Mr. Smith')
print(match.group(1))                       # Smith
print(match.groups())                       # ('Smith',) - only one group

# Backreferences \1, \2, etc
pattern = r'(\w+) \1'                       # Match repeated words
print(re.findall(pattern, 'hello hello world'))  # ['hello']

# Named backreferences (?P=name)
pattern = r'(?P<word>\w+) (?P=word)'
print(re.findall(pattern, 'hello hello world'))  # ['hello']
```

### Alternation and Lookahead

```python
# | - alternation (OR)
pattern = r'cat|dog'
print(re.findall(pattern, 'I have a cat and a dog'))  # ['cat', 'dog']

# With groups
pattern = r'(Mr|Mrs|Ms)\. \w+'
print(re.findall(pattern, 'Mr. Smith and Mrs. Jones'))  # ['Mr', 'Mrs']

# Positive lookahead (?=...)
pattern = r'\d+(?= dollars)'
print(re.findall(pattern, '100 dollars and 50 cents'))  # ['100']

# Negative lookahead (?!...)
pattern = r'\d+(?! dollars)'
print(re.findall(pattern, '100 dollars and 50 cents'))  # ['10', '0', '5', '0']

# Positive lookbehind (?<=...)
pattern = r'(?<=\$)\d+'
print(re.findall(pattern, 'Price: $100'))   # ['100']

# Negative lookbehind (?<!...)
pattern = r'(?<!\$)\d+'
print(re.findall(pattern, 'Price: $100 for 2 items'))  # ['10', '0', '2']
```

## Flags and Options

### Common Flags

```python
# re.IGNORECASE (re.I) - case insensitive
pattern = r'hello'
print(re.search(pattern, 'HELLO', re.IGNORECASE))  # <re.Match object>

# re.MULTILINE (re.M) - ^ and $ match line boundaries
text = 'First line\nSecond line\nThird line'
print(re.findall(r'^Second', text, re.MULTILINE))  # ['Second']

# re.DOTALL (re.S) - . matches newline
text = 'Hello\nWorld'
print(re.search(r'Hello.World', text))      # None
print(re.search(r'Hello.World', text, re.DOTALL))  # <re.Match object>

# re.VERBOSE (re.X) - allow comments and whitespace
pattern = r'''
    \d{3}    # Area code
    -        # Separator
    \d{4}    # Number
'''
print(re.search(pattern, '123-4567', re.VERBOSE))  # <re.Match object>

# Combining flags with |
pattern = r'hello'
print(re.search(pattern, 'HELLO\nWORLD', re.I | re.M))  # <re.Match object>

# Inline flags (?i), (?m), (?s), (?x)
pattern = r'(?i)hello'
print(re.search(pattern, 'HELLO'))          # <re.Match object>

# Multiple inline flags
pattern = r'(?im)^hello'
print(re.search(pattern, 'Line1\nHELLO'))   # <re.Match object>
```

## Compiled Patterns

### Compiling and Reusing Patterns

```python
# Compile pattern for reuse
pattern = re.compile(r'\d+')
print(pattern.findall('Room 123'))          # ['123']
print(pattern.search('Number: 456'))        # <re.Match object>

# Compile with flags
pattern = re.compile(r'hello', re.IGNORECASE)
print(pattern.search('HELLO'))              # <re.Match object>

# More efficient for repeated use
pattern = re.compile(r'\b\w+\b')
text1 = 'Hello World'
text2 = 'Python Regex'
print(pattern.findall(text1))               # ['Hello', 'World']
print(pattern.findall(text2))               # ['Python', 'Regex']

# Access pattern attributes
pattern = re.compile(r'\d+', re.IGNORECASE)
print(pattern.pattern)                      # \d+
print(pattern.flags)                        # 34 (flag value)
```

## Match Objects

### Working with Match Objects

```python
pattern = r'(\d{3})-(\d{4})'
text = 'Phone: 123-4567'
match = re.search(pattern, text)

# Get matched string
print(match.group())                        # 123-4567
print(match.group(0))                       # 123-4567 (same)

# Get captured groups
print(match.group(1))                       # 123
print(match.group(2))                       # 4567
print(match.groups())                       # ('123', '4567')

# Get match position
print(match.start())                        # 7
print(match.end())                          # 15
print(match.span())                         # (7, 15)

# Get group positions
print(match.start(1))                       # 7
print(match.end(1))                         # 10
print(match.span(1))                        # (7, 10)

# Get original string
print(match.string)                         # Phone: 123-4567

# Get regex pattern
print(match.re.pattern)                     # (\d{3})-(\d{4})

# Named groups
pattern = r'(?P<area>\d{3})-(?P<number>\d{4})'
match = re.search(pattern, 'Phone: 123-4567')
print(match.group('area'))                  # 123
print(match.group('number'))                # 4567
print(match.groupdict())                    # {'area': '123', 'number': '4567'}

# Expand template
template = r'Area: \1, Number: \2'
print(match.expand(template))               # Area: 123, Number: 4567
```

## Common Patterns

### Email Validation

```python
# Simple email pattern
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
text = 'Contact: john.doe@example.com'
print(re.findall(pattern, text))            # ['john.doe@example.com']

# Extract all emails from text
text = 'Emails: alice@test.com, bob@example.org'
emails = re.findall(pattern, text)
print(emails)                               # ['alice@test.com', 'bob@example.org']
```

### URL Matching

```python
# Simple URL pattern
pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&/=]*'
text = 'Visit https://www.example.com or http://test.org'
urls = re.findall(pattern, text)
print(urls)                                 # ['https://www.example.com', 'http://test.org']

# Extract domain from URL
pattern = r'https?://(?:www\.)?([^/]+)'
match = re.search(pattern, 'https://www.example.com/page')
print(match.group(1))                       # example.com
```

### Phone Numbers

```python
# US phone numbers
pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
text = 'Call: (123) 456-7890 or 123.456.7890 or 1234567890'
phones = re.findall(pattern, text)
print(phones)                               # ['(123) 456-7890', '123.456.7890', '1234567890']

# Normalize phone numbers
def normalize_phone(phone):
    digits = re.sub(r'\D', '', phone)
    return f'({digits[:3]}) {digits[3:6]}-{digits[6:]}'

print(normalize_phone('1234567890'))        # (123) 456-7890
```

### Date Formats

```python
# Date in YYYY-MM-DD format
pattern = r'\b\d{4}-\d{2}-\d{2}\b'
text = 'Dates: 2024-01-15 and 2024-12-31'
dates = re.findall(pattern, text)
print(dates)                                # ['2024-01-15', '2024-12-31']

# Date in MM/DD/YYYY format
pattern = r'\b\d{2}/\d{2}/\d{4}\b'
text = 'Dates: 01/15/2024 and 12/31/2024'
dates = re.findall(pattern, text)
print(dates)                                # ['01/15/2024', '12/31/2024']

# Extract date components
pattern = r'(?P<month>\d{2})/(?P<day>\d{2})/(?P<year>\d{4})'
match = re.search(pattern, 'Date: 01/15/2024')
print(match.groupdict())                    # {'month': '01', 'day': '15', 'year': '2024'}
```

### IP Addresses

```python
# Simple IP address
pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
text = 'Server: 192.168.1.1 and 10.0.0.1'
ips = re.findall(pattern, text)
print(ips)                                  # ['192.168.1.1', '10.0.0.1']

# Validate IP address (0-255 for each octet)
pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
print(re.search(pattern, '192.168.1.1'))    # <re.Match object>
print(re.search(pattern, '999.999.999.999'))  # None
```

### HTML/XML Tags

```python
# Extract HTML tags
pattern = r'<(\w+)>(.*?)</\1>'
html = '<div>Hello</div><p>World</p>'
tags = re.findall(pattern, html)
print(tags)                                 # [('div', 'Hello'), ('p', 'World')]

# Remove HTML tags
pattern = r'<[^>]+>'
html = '<p>Hello <b>World</b></p>'
text = re.sub(pattern, '', html)
print(text)                                 # Hello World

# Extract tag attributes
pattern = r'<(\w+)\s+(\w+)="([^"]+)">'
html = '<img src="photo.jpg" alt="Photo">'
match = re.search(pattern, html)
print(match.groups())                       # ('img', 'src', 'photo.jpg')
```

### Credit Card Numbers

```python
# Visa, MasterCard, etc.
patterns = {
    'Visa': r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'MasterCard': r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'AmEx': r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'
}

text = 'Cards: 4111-1111-1111-1111, 5500-0000-0000-0004'
for card_type, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if matches:
        print(f'{card_type}: {matches}')
```

## Advanced Techniques

### Substitution with Functions

```python
# Replace using function
def title_case(match):
    return match.group().capitalize()

text = 'hello world python regex'
result = re.sub(r'\b\w+\b', title_case, text)
print(result)                               # Hello World Python Regex

# Replace with counter
counter = 0
def number_words(match):
    global counter
    counter += 1
    return f'{counter}. {match.group()}'

text = 'apple banana cherry'
result = re.sub(r'\w+', number_words, text)
print(result)                               # 1. apple 2. banana 3. cherry

# Use match information in replacement
def format_phone(match):
    area = match.group(1)
    prefix = match.group(2)
    number = match.group(3)
    return f'({area}) {prefix}-{number}'

pattern = r'(\d{3})(\d{3})(\d{4})'
text = '1234567890'
result = re.sub(pattern, format_phone, text)
print(result)                               # (123) 456-7890
```

### Splitting with Groups

```python
# Keep delimiters
text = 'apple,banana;cherry:date'
parts = re.split(r'([,;:])', text)
print(parts)                                # ['apple', ',', 'banana', ';', 'cherry', ':', 'date']

# Process pairs
pairs = zip(parts[::2], parts[1::2])
for item, sep in pairs:
    print(f'{item} separated by {sep}')
```

### Conditional Patterns

```python
# Match different formats
pattern = r'\d{3}-\d{4}|\d{10}|\(\d{3}\)\s?\d{3}-\d{4}'
phones = ['123-4567', '1234567890', '(123) 456-7890']
for phone in phones:
    match = re.search(pattern, phone)
    if match:
        print(f'Valid: {match.group()}')

# Named groups with alternation
pattern = r'(?P<format1>\d{3}-\d{4})|(?P<format2>\d{10})'
match = re.search(pattern, '123-4567')
print(match.lastgroup)                      # format1
```

### Scanning

```python
# Lexical scanning
token_pattern = r'''
    (?P<NUMBER>\d+)|
    (?P<PLUS>\+)|
    (?P<MINUS>-)|
    (?P<TIMES>\*)|
    (?P<DIVIDE>/)|
    (?P<WHITESPACE>\s+)
'''
pattern = re.compile(token_pattern, re.VERBOSE)

text = '10 + 20 * 3'
for match in pattern.finditer(text):
    kind = match.lastgroup
    value = match.group()
    if kind != 'WHITESPACE':
        print(f'{kind}: {value}')
# Output:
# NUMBER: 10
# PLUS: +
# NUMBER: 20
# TIMES: *
# NUMBER: 3
```

## Performance Tips

```python
# Compile patterns for repeated use
# Bad
for text in large_list:
    if re.search(r'\d+', text):
        process(text)

# Good
pattern = re.compile(r'\d+')
for text in large_list:
    if pattern.search(text):
        process(text)

# Use raw strings for patterns
# Bad
pattern = '\\d+\\s+\\w+'                    # Hard to read

# Good
pattern = r'\d+\s+\w+'                      # Clear and correct

# Be specific with patterns
# Bad (slow)
pattern = r'.*foo.*'                        # Too greedy

# Good (faster)
pattern = r'\bfoo\b'                        # More specific

# Use non-capturing groups when possible
# Bad
pattern = r'(Mr|Mrs|Ms)\. (\w+) (and|or) (\w+)'

# Good (if you don't need first and third groups)
pattern = r'(?:Mr|Mrs|Ms)\. (\w+) (?:and|or) (\w+)'
```

## Debugging Regex

```python
# Test patterns interactively
import re

def test_pattern(pattern, text):
    print(f"Pattern: {pattern}")
    print(f"Text: {text}")
    match = re.search(pattern, text)
    if match:
        print(f"Match: {match.group()}")
        print(f"Groups: {match.groups()}")
        print(f"Span: {match.span()}")
    else:
        print("No match")
    print()

test_pattern(r'\d+', 'abc123def')

# Use online regex testers
# - regex101.com
# - regexr.com
# - pythex.org

# Verbose mode for complex patterns
pattern = r'''
    ^                   # Start of string
    (?P<username>       # Username group
        [a-zA-Z0-9_-]{3,16}  # 3-16 chars
    )
    @                   # Literal @
    (?P<domain>         # Domain group
        [a-zA-Z0-9.-]+  # Domain name
        \.[a-zA-Z]{2,}  # TLD
    )
    $                   # End of string
'''
email_pattern = re.compile(pattern, re.VERBOSE)
```

---

*This document covers comprehensive regular expression usage in Python. For the most up-to-date information, refer to the official Python documentation and the `re` module reference.*
