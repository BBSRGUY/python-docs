# Python Web Development and Internet Programming

This document provides a comprehensive guide to Python web development, internet programming, and HTTP operations with syntax and usage examples.

## Built-in HTTP and URL Libraries

### `urllib` Module

#### `urllib.request` - Opening URLs
```python
import urllib.request
import urllib.parse
import urllib.error

# Basic URL fetching
def fetch_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read()
            encoding = response.info().get_content_charset('utf-8')
            return content.decode(encoding)
    except urllib.error.URLError as e:
        print(f"Error fetching URL: {e}")
        return None

# GET request with parameters
def get_with_params(base_url, params):
    query_string = urllib.parse.urlencode(params)
    full_url = f"{base_url}?{query_string}"
    return fetch_url(full_url)

# POST request
def post_request(url, data):
    data_encoded = urllib.parse.urlencode(data).encode('utf-8')
    req = urllib.request.Request(url, data=data_encoded, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            return response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(f"POST request failed: {e}")
        return None

# Custom headers
def request_with_headers(url, headers=None):
    if headers is None:
        headers = {'User-Agent': 'Python/3.x'}
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"Status: {response.status}")
            print(f"Headers: {dict(response.headers)}")
            return response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(f"Request failed: {e}")
        return None

# Example usage
# content = fetch_url("https://httpbin.org/get")
# result = get_with_params("https://httpbin.org/get", {"key": "value", "name": "test"})
# post_result = post_request("https://httpbin.org/post", {"username": "user", "password": "pass"})
```

#### `urllib.parse` - URL Parsing
```python
import urllib.parse

# Parse URLs
def parse_url_demo():
    url = "https://www.example.com:8080/path/to/page?param1=value1&param2=value2#section"
    
    parsed = urllib.parse.urlparse(url)
    print(f"Scheme: {parsed.scheme}")           # https
    print(f"Netloc: {parsed.netloc}")           # www.example.com:8080
    print(f"Path: {parsed.path}")               # /path/to/page
    print(f"Query: {parsed.query}")             # param1=value1&param2=value2
    print(f"Fragment: {parsed.fragment}")       # section
    
    # Parse query parameters
    query_params = urllib.parse.parse_qs(parsed.query)
    print(f"Query params: {query_params}")      # {'param1': ['value1'], 'param2': ['value2']}
    
    return parsed

# Build URLs
def build_url(scheme="https", netloc="api.example.com", path="/v1/users", 
              params=None, query=None, fragment=None):
    if params is None:
        params = {}
    if query is None:
        query = {}
    
    query_string = urllib.parse.urlencode(query)
    
    components = urllib.parse.ParseResult(
        scheme=scheme,
        netloc=netloc,
        path=path,
        params=urllib.parse.urlencode(params),
        query=query_string,
        fragment=fragment or ""
    )
    
    return urllib.parse.urlunparse(components)

# URL encoding/decoding
def url_encoding_demo():
    # Encode special characters
    text = "Hello World! How are you?"
    encoded = urllib.parse.quote(text)
    print(f"Encoded: {encoded}")                # Hello%20World%21%20How%20are%20you%3F
    
    # Decode
    decoded = urllib.parse.unquote(encoded)
    print(f"Decoded: {decoded}")                # Hello World! How are you?
    
    # Encode for form data
    form_data = {"name": "John Doe", "email": "john@example.com"}
    encoded_form = urllib.parse.urlencode(form_data)
    print(f"Form encoded: {encoded_form}")      # name=John+Doe&email=john%40example.com

# Join URLs
def join_urls(base, *parts):
    """Join URL parts properly"""
    url = base
    for part in parts:
        url = urllib.parse.urljoin(url, part)
    return url

# Example
# base_url = "https://api.example.com/v1/"
# full_url = join_urls(base_url, "users", "123", "profile")
# print(full_url)  # https://api.example.com/v1/users/123/profile
```

### `http` Module

#### `http.client` - Low-level HTTP Client
```python
import http.client
import json

class HTTPClient:
    def __init__(self, host, port=None, use_ssl=False):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.connection = None
    
    def connect(self):
        if self.use_ssl:
            self.connection = http.client.HTTPSConnection(self.host, self.port)
        else:
            self.connection = http.client.HTTPConnection(self.host, self.port)
    
    def request(self, method, path, body=None, headers=None):
        if headers is None:
            headers = {}
        
        if not self.connection:
            self.connect()
        
        try:
            self.connection.request(method, path, body, headers)
            response = self.connection.getresponse()
            
            return {
                'status': response.status,
                'reason': response.reason,
                'headers': dict(response.getheaders()),
                'body': response.read().decode('utf-8')
            }
        except Exception as e:
            return {'error': str(e)}
        finally:
            if self.connection:
                self.connection.close()
                self.connection = None
    
    def get(self, path, headers=None):
        return self.request('GET', path, headers=headers)
    
    def post(self, path, data=None, headers=None):
        if headers is None:
            headers = {}
        
        if isinstance(data, dict):
            data = json.dumps(data)
            headers['Content-Type'] = 'application/json'
        
        return self.request('POST', path, body=data, headers=headers)

# Example usage
# client = HTTPClient('httpbin.org', use_ssl=True)
# response = client.get('/get')
# print(response)
```

#### `http.server` - Simple HTTP Server
```python
import http.server
import socketserver
import threading
import json
from urllib.parse import urlparse, parse_qs

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        
        if parsed_path.path == '/api/hello':
            self.send_json_response({'message': 'Hello, World!', 'params': query_params})
        elif parsed_path.path == '/api/status':
            self.send_json_response({'status': 'OK', 'server': 'Python HTTP Server'})
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        if self.path == '/api/echo':
            try:
                data = json.loads(post_data)
                response = {'received': data, 'method': 'POST'}
                self.send_json_response(response)
            except json.JSONDecodeError:
                self.send_error_response(400, 'Invalid JSON')
        else:
            self.send_error_response(404, 'Not Found')
    
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode('utf-8'))
    
    def send_error_response(self, status, message):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_data = json.dumps({'error': message})
        self.wfile.write(error_data.encode('utf-8'))

def start_server(port=8000, directory='.'):
    """Start a simple HTTP server"""
    import os
    os.chdir(directory)
    
    with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
        print(f"Server running on http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
            httpd.shutdown()

# Example usage
# start_server(8080)
```

## Requests Library

### Basic Requests Operations
```python
import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Basic GET request
def simple_get(url, params=None, headers=None):
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json() if 'application/json' in response.headers.get('content-type', '') else response.text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# POST request with JSON data
def post_json(url, data, headers=None):
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"POST request failed: {e}")
        return None

# File upload
def upload_file(url, file_path, field_name='file'):
    try:
        with open(file_path, 'rb') as f:
            files = {field_name: f}
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"File upload failed: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Session with authentication
class APIClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Python API Client/1.0',
            'Accept': 'application/json'
        })
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get(self, endpoint, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.get(url, **kwargs)
    
    def post(self, endpoint, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.post(url, **kwargs)
    
    def put(self, endpoint, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.put(url, **kwargs)
    
    def delete(self, endpoint, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.delete(url, **kwargs)

# Advanced request with timeout and proxies
def advanced_request(url, method='GET', **kwargs):
    """Make request with advanced options"""
    defaults = {
        'timeout': (5, 30),  # (connect timeout, read timeout)
        'allow_redirects': True,
        'verify': True,  # SSL verification
    }
    
    # Merge defaults with provided kwargs
    options = {**defaults, **kwargs}
    
    try:
        response = requests.request(method, url, **options)
        response.raise_for_status()
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.text,
            'json': response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
    except requests.exceptions.Timeout:
        return {'error': 'Request timed out'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Connection failed'}
    except requests.exceptions.HTTPError as e:
        return {'error': f'HTTP error: {e}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Request failed: {e}'}

# Example usage
# client = APIClient('https://jsonplaceholder.typicode.com', api_key='your-api-key')
# users = client.get('/users')
# print(users.json())
```

### Requests Sessions and Cookies
```python
import requests
from requests.cookies import RequestsCookieJar

class WebSession:
    def __init__(self):
        self.session = requests.Session()
        self.login_status = False
    
    def login(self, login_url, username, password, csrf_token=None):
        """Perform login and maintain session"""
        # Get login page first (to get CSRF token if needed)
        login_page = self.session.get(login_url)
        
        # Extract CSRF token if needed
        if csrf_token is None and 'csrf' in login_page.text.lower():
            # Simple CSRF token extraction (would need proper parsing in real app)
            import re
            csrf_match = re.search(r'csrf["\']?\s*[:=]\s*["\']([^"\']+)', login_page.text)
            if csrf_match:
                csrf_token = csrf_match.group(1)
        
        # Prepare login data
        login_data = {
            'username': username,
            'password': password
        }
        
        if csrf_token:
            login_data['csrf_token'] = csrf_token
        
        # Perform login
        response = self.session.post(login_url, data=login_data)
        
        # Check if login was successful
        if response.status_code == 200 and 'error' not in response.text.lower():
            self.login_status = True
            return True
        else:
            return False
    
    def get_protected_resource(self, url):
        """Access protected resource using session cookies"""
        if not self.login_status:
            raise Exception("Not logged in")
        
        response = self.session.get(url)
        return response
    
    def save_cookies(self, filename):
        """Save session cookies to file"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.session.cookies, f)
    
    def load_cookies(self, filename):
        """Load session cookies from file"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                self.session.cookies = pickle.load(f)
                self.login_status = True
                return True
        except FileNotFoundError:
            return False

# Custom cookie handling
def cookie_demo():
    # Create custom cookie jar
    jar = RequestsCookieJar()
    jar.set('session_id', 'abc123', domain='example.com')
    jar.set('user_pref', 'dark_mode', domain='example.com')
    
    # Use cookies in request
    response = requests.get('https://httpbin.org/cookies', cookies=jar)
    print("Sent cookies:", response.request.headers.get('Cookie'))
    
    # Extract cookies from response
    response = requests.get('https://httpbin.org/cookies/set/test_cookie/test_value')
    print("Received cookies:", response.cookies)
    
    return jar
```

## Web Scraping

### BeautifulSoup for HTML Parsing
```python
# pip install beautifulsoup4 lxml
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin, urlparse

class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_page(self, url):
        """Fetch and parse a web page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_links(self, soup, filter_func=None):
        """Extract all links from a page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            full_url = urljoin(self.base_url, href)
            
            if filter_func is None or filter_func(full_url):
                links.append({
                    'url': full_url,
                    'text': link.get_text(strip=True),
                    'title': link.get('title', '')
                })
        
        return links
    
    def extract_images(self, soup):
        """Extract all images from a page"""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                full_url = urljoin(self.base_url, src)
                images.append({
                    'url': full_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width'),
                    'height': img.get('height')
                })
        
        return images
    
    def extract_text_content(self, soup, selectors=None):
        """Extract text content using CSS selectors"""
        if selectors is None:
            selectors = ['h1', 'h2', 'h3', 'p', 'li']
        
        content = {}
        for selector in selectors:
            elements = soup.select(selector)
            content[selector] = [elem.get_text(strip=True) for elem in elements]
        
        return content
    
    def extract_forms(self, soup):
        """Extract form information"""
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'GET').upper(),
                'fields': []
            }
            
            # Extract form fields
            for field in form.find_all(['input', 'textarea', 'select']):
                field_info = {
                    'name': field.get('name', ''),
                    'type': field.get('type', 'text'),
                    'value': field.get('value', ''),
                    'required': field.has_attr('required')
                }
                form_data['fields'].append(field_info)
            
            forms.append(form_data)
        
        return forms

# Advanced scraping techniques
def scrape_with_pagination(scraper, start_url, max_pages=10):
    """Scrape multiple pages with pagination"""
    all_data = []
    current_url = start_url
    page_count = 0
    
    while current_url and page_count < max_pages:
        print(f"Scraping page {page_count + 1}: {current_url}")
        soup = scraper.get_page(current_url)
        
        if not soup:
            break
        
        # Extract data from current page
        page_data = scraper.extract_text_content(soup)
        all_data.append(page_data)
        
        # Find next page link
        next_link = soup.find('a', string=re.compile(r'Next|→|>'))
        if next_link and next_link.get('href'):
            current_url = urljoin(current_url, next_link['href'])
        else:
            break
        
        page_count += 1
    
    return all_data

# Table scraping
def scrape_table(soup, table_selector='table'):
    """Extract data from HTML tables"""
    tables = []
    
    for table in soup.select(table_selector):
        table_data = {
            'headers': [],
            'rows': []
        }
        
        # Extract headers
        header_row = table.find('tr')
        if header_row:
            headers = header_row.find_all(['th', 'td'])
            table_data['headers'] = [h.get_text(strip=True) for h in headers]
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:  # Only add non-empty rows
                table_data['rows'].append(row_data)
        
        tables.append(table_data)
    
    return tables

# Example usage
# scraper = WebScraper('https://example.com')
# soup = scraper.get_page('https://example.com/page.html')
# links = scraper.extract_links(soup)
# content = scraper.extract_text_content(soup)
```

### Selenium for Dynamic Content
```python
# pip install selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

class SeleniumScraper:
    def __init__(self, headless=True, implicit_wait=10):
        self.options = Options()
        if headless:
            self.options.add_argument('--headless')
        
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.implicitly_wait(implicit_wait)
        self.wait = WebDriverWait(self.driver, 20)
    
    def get_page(self, url):
        """Navigate to a page and wait for it to load"""
        self.driver.get(url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    
    def wait_for_element(self, selector, by=By.CSS_SELECTOR, timeout=20):
        """Wait for element to be present"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            print(f"Element {selector} not found within {timeout} seconds")
            return None
    
    def click_element(self, selector, by=By.CSS_SELECTOR):
        """Click an element"""
        try:
            element = self.wait.until(EC.element_to_be_clickable((by, selector)))
            element.click()
            return True
        except TimeoutException:
            print(f"Element {selector} not clickable")
            return False
    
    def fill_form(self, form_data):
        """Fill out a form with provided data"""
        for field_name, value in form_data.items():
            try:
                field = self.driver.find_element(By.NAME, field_name)
                field.clear()
                field.send_keys(value)
            except NoSuchElementException:
                print(f"Field {field_name} not found")
    
    def scroll_to_load_content(self, pause_time=2, max_scrolls=10):
        """Scroll down to load dynamic content"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scrolls = 0
        
        while scrolls < max_scrolls:
            # Scroll to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            
            # Check if new content loaded
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            
            last_height = new_height
            scrolls += 1
    
    def extract_dynamic_content(self, container_selector, item_selector):
        """Extract content that loads dynamically"""
        # Wait for container to load
        container = self.wait_for_element(container_selector)
        if not container:
            return []
        
        # Scroll to load all content
        self.scroll_to_load_content()
        
        # Extract items
        items = self.driver.find_elements(By.CSS_SELECTOR, item_selector)
        return [item.text for item in items]
    
    def take_screenshot(self, filename):
        """Take a screenshot of the current page"""
        self.driver.save_screenshot(filename)
    
    def execute_javascript(self, script):
        """Execute JavaScript code"""
        return self.driver.execute_script(script)
    
    def close(self):
        """Close the browser"""
        self.driver.quit()

# Example usage for scraping a dynamic website
def scrape_dynamic_site(url):
    scraper = SeleniumScraper(headless=True)
    
    try:
        # Navigate to the page
        scraper.get_page(url)
        
        # Wait for specific content to load
        scraper.wait_for_element('#content')
        
        # Click on a button to load more content
        scraper.click_element('button[data-load-more]')
        
        # Extract dynamic content
        items = scraper.extract_dynamic_content('.container', '.item')
        
        # Take a screenshot
        scraper.take_screenshot('page_screenshot.png')
        
        return items
    
    finally:
        scraper.close()

# Handle cookies and local storage
def handle_browser_storage(scraper):
    # Get cookies
    cookies = scraper.driver.get_cookies()
    print("Cookies:", cookies)
    
    # Add a cookie
    scraper.driver.add_cookie({'name': 'test', 'value': 'test_value'})
    
    # Local storage operations
    scraper.driver.execute_script("localStorage.setItem('key', 'value');")
    value = scraper.driver.execute_script("return localStorage.getItem('key');")
    print("Local storage value:", value)
```

## Flask Web Framework

### Basic Flask Application
```python
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask import send_file, abort, make_response
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Basic routes
@app.route('/')
def home():
    return '<h1>Welcome to Flask!</h1>'

@app.route('/hello/<name>')
def hello(name):
    return f'<h1>Hello, {name}!</h1>'

# HTTP methods
@app.route('/api/users', methods=['GET', 'POST'])
def users():
    if request.method == 'GET':
        # Return all users
        users = [
            {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
            {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
        ]
        return jsonify(users)
    
    elif request.method == 'POST':
        # Create new user
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Name is required'}), 400
        
        new_user = {
            'id': 3,
            'name': data['name'],
            'email': data.get('email', '')
        }
        return jsonify(new_user), 201

@app.route('/api/users/<int:user_id>', methods=['GET', 'PUT', 'DELETE'])
def user_detail(user_id):
    if request.method == 'GET':
        # Get specific user
        user = {'id': user_id, 'name': f'User {user_id}'}
        return jsonify(user)
    
    elif request.method == 'PUT':
        # Update user
        data = request.get_json()
        return jsonify({'id': user_id, 'updated': True, 'data': data})
    
    elif request.method == 'DELETE':
        # Delete user
        return jsonify({'id': user_id, 'deleted': True})

# Query parameters
@app.route('/search')
def search():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    return jsonify({
        'query': query,
        'page': page,
        'per_page': per_page,
        'results': []
    })

# Form handling
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        return '''
        <form method="POST">
            <input type="text" name="name" placeholder="Name" required>
            <input type="email" name="email" placeholder="Email" required>
            <textarea name="message" placeholder="Message" required></textarea>
            <button type="submit">Send</button>
        </form>
        '''
    else:
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Process form data
        return f'Thank you {name}! Your message has been received.'

# File upload
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return '''
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button type="submit">Upload</button>
        </form>
        '''
    else:
        if 'file' not in request.files:
            return 'No file selected', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        
        if file:
            filename = file.filename
            file.save(os.path.join('uploads', filename))
            return f'File {filename} uploaded successfully!'

# Cookies and sessions
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return '''
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        '''
    else:
        username = request.form['username']
        password = request.form['password']
        
        # Simple authentication (use proper auth in production)
        if username == 'admin' and password == 'secret':
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid credentials', 401

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    return f'Welcome to the dashboard, {session["user"]}!'

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Before/after request hooks
@app.before_request
def before_request():
    # Log request info
    print(f"{datetime.now()}: {request.method} {request.url}")

@app.after_request
def after_request(response):
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response

if __name__ == '__main__':
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Flask with Templates and Database
```python
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database models
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Post {self.title}>'

# Routes
@app.route('/')
def index():
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template('index.html', posts=posts)

@app.route('/post/<int:id>')
def post_detail(id):
    post = Post.query.get_or_404(id)
    return render_template('post.html', post=post)

@app.route('/create', methods=['GET', 'POST'])
def create_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        
        if title and content:
            post = Post(title=title, content=content)
            db.session.add(post)
            db.session.commit()
            flash('Post created successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Please fill in all fields', 'error')
    
    return render_template('create.html')

# Create tables
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)

# Template files would be:
# templates/base.html
base_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Flask Blog{% endblock %}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .post { border: 1px solid #ddd; padding: 20px; margin: 20px 0; }
        .flash { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        <a href="{{ url_for('create_post') }}">Create Post</a>
    </nav>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="flash {{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    {% block content %}{% endblock %}
</body>
</html>
'''

# templates/index.html
index_template = '''
{% extends "base.html" %}

{% block content %}
<h1>Blog Posts</h1>

{% for post in posts %}
<div class="post">
    <h2><a href="{{ url_for('post_detail', id=post.id) }}">{{ post.title }}</a></h2>
    <p>{{ post.content[:200] }}...</p>
    <small>{{ post.created_at.strftime('%Y-%m-%d %H:%M') }}</small>
</div>
{% endfor %}
{% endblock %}
'''
```

## Django Web Framework

### Basic Django Project Structure
```python
# settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',  # Your app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    class Meta:
        verbose_name_plural = "categories"
    
    def __str__(self):
        return self.name

class Post(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('post_detail', kwargs={'slug': self.slug})

# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Post, Category
from .forms import PostForm

def post_list(request):
    posts = Post.objects.filter(published=True)
    
    # Search functionality
    query = request.GET.get('q')
    if query:
        posts = posts.filter(
            Q(title__icontains=query) | Q(content__icontains=query)
        )
    
    # Category filter
    category_id = request.GET.get('category')
    if category_id:
        posts = posts.filter(category_id=category_id)
    
    # Pagination
    paginator = Paginator(posts, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    categories = Category.objects.all()
    
    context = {
        'page_obj': page_obj,
        'categories': categories,
        'query': query,
        'selected_category': category_id
    }
    return render(request, 'blog/post_list.html', context)

def post_detail(request, slug):
    post = get_object_or_404(Post, slug=slug, published=True)
    
    # Related posts
    related_posts = Post.objects.filter(
        category=post.category,
        published=True
    ).exclude(id=post.id)[:3]
    
    context = {
        'post': post,
        'related_posts': related_posts
    }
    return render(request, 'blog/post_detail.html', context)

@login_required
def create_post(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            messages.success(request, 'Post created successfully!')
            return redirect(post.get_absolute_url())
    else:
        form = PostForm()
    
    return render(request, 'blog/create_post.html', {'form': form})

# API views
def api_posts(request):
    posts = Post.objects.filter(published=True).values(
        'id', 'title', 'slug', 'created_at'
    )
    return JsonResponse(list(posts), safe=False)

def api_post_detail(request, slug):
    post = get_object_or_404(Post, slug=slug, published=True)
    data = {
        'id': post.id,
        'title': post.title,
        'content': post.content,
        'author': post.author.username,
        'category': post.category.name if post.category else None,
        'created_at': post.created_at.isoformat()
    }
    return JsonResponse(data)

# forms.py
from django import forms
from .models import Post, Category

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'slug', 'content', 'category', 'published']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'slug': forms.TextInput(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 10}),
            'category': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def clean_slug(self):
        slug = self.cleaned_data['slug']
        if Post.objects.filter(slug=slug).exists():
            raise forms.ValidationError("A post with this slug already exists.")
        return slug

# urls.py
from django.urls import path
from . import views

app_name = 'blog'

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('post/<slug:slug>/', views.post_detail, name='post_detail'),
    path('create/', views.create_post, name='create_post'),
    
    # API endpoints
    path('api/posts/', views.api_posts, name='api_posts'),
    path('api/posts/<slug:slug>/', views.api_post_detail, name='api_post_detail'),
]
```

## FastAPI Framework

### FastAPI Application
```python
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uvicorn
from datetime import datetime, timedelta
import jwt

app = FastAPI(title="Blog API", version="1.0.0")

# Security
security = HTTPBearer()
SECRET_KEY = "your-secret-key"

# Pydantic models
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class PostBase(BaseModel):
    title: str
    content: str

class PostCreate(PostBase):
    pass

class Post(PostBase):
    id: int
    author_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

# Mock database
fake_users_db = {}
fake_posts_db = {}

# Authentication
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(username: str = Depends(verify_token)):
    user = fake_users_db.get(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Blog API"}

@app.post("/register", response_model=User)
async def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    fake_user = {
        "id": len(fake_users_db) + 1,
        "username": user.username,
        "email": user.email,
        "password": user.password,  # In production, hash this!
        "created_at": datetime.utcnow()
    }
    fake_users_db[user.username] = fake_user
    return fake_user

@app.post("/login", response_model=Token)
async def login(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/posts", response_model=List[Post])
async def get_posts(skip: int = 0, limit: int = 10):
    posts = list(fake_posts_db.values())[skip:skip + limit]
    return posts

@app.post("/posts", response_model=Post)
async def create_post(post: PostCreate, current_user: dict = Depends(get_current_user)):
    fake_post = {
        "id": len(fake_posts_db) + 1,
        "title": post.title,
        "content": post.content,
        "author_id": current_user["id"],
        "created_at": datetime.utcnow(),
        "updated_at": None
    }
    fake_posts_db[fake_post["id"]] = fake_post
    return fake_post

@app.get("/posts/{post_id}", response_model=Post)
async def get_post(post_id: int):
    post = fake_posts_db.get(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.put("/posts/{post_id}", response_model=Post)
async def update_post(post_id: int, post: PostCreate, current_user: dict = Depends(get_current_user)):
    existing_post = fake_posts_db.get(post_id)
    if not existing_post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if existing_post["author_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to update this post")
    
    existing_post.update({
        "title": post.title,
        "content": post.content,
        "updated_at": datetime.utcnow()
    })
    return existing_post

@app.delete("/posts/{post_id}")
async def delete_post(post_id: int, current_user: dict = Depends(get_current_user)):
    post = fake_posts_db.get(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if post["author_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this post")
    
    del fake_posts_db[post_id]
    return {"message": "Post deleted successfully"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## WebSockets

### WebSocket Server and Client
```python
# WebSocket server with asyncio
import asyncio
import websockets
import json
from datetime import datetime

class WebSocketServer:
    def __init__(self):
        self.clients = set()
        self.rooms = {}
    
    async def register(self, websocket, path):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            await self.handle_client(websocket)
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket):
        """Handle messages from a client"""
        async for message in websocket:
            try:
                data = json.loads(message)
                await self.process_message(websocket, data)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'error': 'Invalid JSON message'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'error': str(e)
                }))
    
    async def process_message(self, websocket, data):
        """Process different types of messages"""
        message_type = data.get('type')
        
        if message_type == 'chat':
            await self.handle_chat_message(websocket, data)
        elif message_type == 'join_room':
            await self.handle_join_room(websocket, data)
        elif message_type == 'leave_room':
            await self.handle_leave_room(websocket, data)
        elif message_type == 'ping':
            await websocket.send(json.dumps({'type': 'pong'}))
        else:
            await websocket.send(json.dumps({
                'error': f'Unknown message type: {message_type}'
            }))
    
    async def handle_chat_message(self, websocket, data):
        """Handle chat messages"""
        room = data.get('room', 'general')
        username = data.get('username', 'Anonymous')
        message = data.get('message', '')
        
        chat_message = {
            'type': 'chat',
            'room': room,
            'username': username,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all clients in the room
        await self.broadcast_to_room(room, chat_message)
    
    async def handle_join_room(self, websocket, data):
        """Handle joining a room"""
        room = data.get('room')
        username = data.get('username')
        
        if room not in self.rooms:
            self.rooms[room] = set()
        
        self.rooms[room].add(websocket)
        
        # Notify room about new user
        notification = {
            'type': 'user_joined',
            'room': room,
            'username': username,
            'timestamp': datetime.now().isoformat()
        }
        await self.broadcast_to_room(room, notification)
    
    async def handle_leave_room(self, websocket, data):
        """Handle leaving a room"""
        room = data.get('room')
        username = data.get('username')
        
        if room in self.rooms and websocket in self.rooms[room]:
            self.rooms[room].remove(websocket)
            
            # Notify room about user leaving
            notification = {
                'type': 'user_left',
                'room': room,
                'username': username,
                'timestamp': datetime.now().isoformat()
            }
            await self.broadcast_to_room(room, notification)
    
    async def broadcast_to_room(self, room, message):
        """Broadcast message to all clients in a room"""
        if room not in self.rooms:
            return
        
        disconnected_clients = []
        for client in self.rooms[room]:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.rooms[room].discard(client)
    
    async def broadcast_to_all(self, message):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)

# Start WebSocket server
async def start_websocket_server():
    server = WebSocketServer()
    start_server = websockets.serve(server.register, "localhost", 8765)
    print("WebSocket server starting on ws://localhost:8765")
    await start_server

# WebSocket client
class WebSocketClient:
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket server"""
        self.websocket = await websockets.connect(self.uri)
        print(f"Connected to {self.uri}")
    
    async def send_message(self, message_type, **kwargs):
        """Send a message to the server"""
        message = {'type': message_type, **kwargs}
        await self.websocket.send(json.dumps(message))
    
    async def listen(self):
        """Listen for messages from server"""
        async for message in self.websocket:
            data = json.loads(message)
            await self.handle_message(data)
    
    async def handle_message(self, data):
        """Handle messages from server"""
        message_type = data.get('type')
        
        if message_type == 'chat':
            print(f"[{data['room']}] {data['username']}: {data['message']}")
        elif message_type == 'user_joined':
            print(f"User {data['username']} joined {data['room']}")
        elif message_type == 'user_left':
            print(f"User {data['username']} left {data['room']}")
        elif message_type == 'pong':
            print("Received pong")
        else:
            print(f"Received: {data}")
    
    async def close(self):
        """Close the connection"""
        if self.websocket:
            await self.websocket.close()

# Example client usage
async def websocket_client_example():
    client = WebSocketClient("ws://localhost:8765")
    
    try:
        await client.connect()
        
        # Join a room
        await client.send_message('join_room', room='general', username='Alice')
        
        # Send a chat message
        await client.send_message('chat', room='general', username='Alice', message='Hello, everyone!')
        
        # Listen for messages
        await client.listen()
    
    except KeyboardInterrupt:
        print("Client stopped")
    finally:
        await client.close()

# Run server
# asyncio.run(start_websocket_server())

# Run client
# asyncio.run(websocket_client_example())
```

### Flask-SocketIO WebSocket Integration
```python
# pip install flask-socketio
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store connected users
connected_users = {}

@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('connect')
def handle_connect():
    user_id = str(uuid.uuid4())
    connected_users[request.sid] = {
        'id': user_id,
        'username': None,
        'rooms': set()
    }
    print(f'Client connected: {request.sid}')
    emit('connected', {'user_id': user_id})

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in connected_users:
        user = connected_users[request.sid]
        username = user.get('username', 'Anonymous')
        
        # Leave all rooms
        for room in user['rooms']:
            leave_room(room)
            emit('user_left', {
                'username': username,
                'room': room
            }, room=room)
        
        del connected_users[request.sid]
    print(f'Client disconnected: {request.sid}')

@socketio.on('set_username')
def handle_set_username(data):
    username = data['username']
    if request.sid in connected_users:
        connected_users[request.sid]['username'] = username
        emit('username_set', {'username': username})

@socketio.on('join_room')
def handle_join_room(data):
    room = data['room']
    username = data.get('username', 'Anonymous')
    
    join_room(room)
    
    if request.sid in connected_users:
        connected_users[request.sid]['rooms'].add(room)
    
    emit('user_joined', {
        'username': username,
        'room': room
    }, room=room)

@socketio.on('leave_room')
def handle_leave_room(data):
    room = data['room']
    username = data.get('username', 'Anonymous')
    
    leave_room(room)
    
    if request.sid in connected_users:
        connected_users[request.sid]['rooms'].discard(room)
    
    emit('user_left', {
        'username': username,
        'room': room
    }, room=room)

@socketio.on('send_message')
def handle_message(data):
    room = data['room']
    username = data['username']
    message = data['message']
    
    emit('message', {
        'username': username,
        'message': message,
        'room': room,
        'timestamp': datetime.now().isoformat()
    }, room=room)

@socketio.on('typing')
def handle_typing(data):
    room = data['room']
    username = data['username']
    
    emit('user_typing', {
        'username': username,
        'room': room
    }, room=room, include_self=False)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

# HTML template (templates/chat.html)
chat_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Chat</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <div id="messages"></div>
    <input type="text" id="messageInput" placeholder="Type a message...">
    <button onclick="sendMessage()">Send</button>
    
    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('message', function(data) {
            const messages = document.getElementById('messages');
            messages.innerHTML += '<p><strong>' + data.username + ':</strong> ' + data.message + '</p>';
        });
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value;
            if (message) {
                socket.emit('send_message', {
                    username: 'User',
                    message: message,
                    room: 'general'
                });
                input.value = '';
            }
        }
        
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Join general room on connect
        socket.on('connected', function(data) {
            socket.emit('join_room', {
                room: 'general',
                username: 'User'
            });
        });
    </script>
</body>
</html>
'''
```

---

*This document covers comprehensive web development and internet programming in Python including built-in HTTP libraries, popular frameworks (Flask, Django, FastAPI), web scraping, WebSocket programming, and various web-related utilities. For the most up-to-date information, refer to the official documentation of the respective frameworks and libraries.*