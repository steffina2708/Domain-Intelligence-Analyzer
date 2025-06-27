# Enhanced Domain Intelligence Analyzer - Flask API Backend
# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup # type: ignore
import time
import re
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer # type: ignore
import chromadb # type: ignore
from typing import List, Dict, Tuple
import os
from datetime import datetime, timedelta
import warnings
import threading
import hashlib
import uuid
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Groq API Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
    GROQ_MODEL = "llama3-8b-8192"

    # Crawling Configuration
    MAX_PAGES = 25
    CRAWL_DELAY = 1
    MAX_CONTENT_LENGTH = 10000
    SYNC_CHECK_INTERVAL = 24

    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Chat Configuration
    MAX_CHAT_HISTORY = 10
    CONTEXT_CHUNKS = 5

# =============================================================================
# ENHANCED WEB CRAWLER CLASS
# =============================================================================
class EnhancedDomainCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited_urls = set()
        self.domain_data = {}
        self.last_crawl_hashes = {}

    def get_page_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_domain)

            if parsed.netloc != base_parsed.netloc:
                return False

            skip_patterns = [
                'login', 'register', 'cart', 'checkout', 'admin',
                'wp-admin', 'wp-content', '.pdf', '.jpg', '.png', '.gif',
                'privacy', 'terms', 'cookie', 'legal', '#', 'javascript:'
            ]

            return not any(pattern in url.lower() for pattern in skip_patterns)
        except:
            return False

    def extract_content(self, html: str, url: str) -> Dict:
        soup = BeautifulSoup(html, 'html.parser')

        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        title = soup.find('title')
        title = title.text.strip() if title else "No Title"

        content_selectors = [
            'main', '[role="main"]', '.main-content', '#main-content',
            '.content', '#content', 'article', '.post', '.page-content'
        ]

        main_content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text(separator=' ', strip=True)
                break

        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)

        main_content = re.sub(r'\s+', ' ', main_content).strip()
        main_content = main_content[:Config.MAX_CONTENT_LENGTH]

        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            headings.append(h.get_text().strip())

        content_hash = self.get_page_hash(main_content)

        return {
            'url': url,
            'title': title,
            'content': main_content,
            'headings': headings,
            'word_count': len(main_content.split()),
            'content_hash': content_hash,
            'timestamp': datetime.now().isoformat()
        }

    def discover_pages(self, domain: str) -> List[str]:
        urls_to_crawl = [domain]

        try:
            response = self.session.get(domain, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            links = []
            for a in soup.find_all('a', href=True):
                url = urljoin(domain, a['href'])
                if self.is_valid_url(url, domain):
                    links.append(url)

            priority_keywords = [
                'about', 'service', 'product', 'solution', 'team',
                'contact', 'portfolio', 'work', 'case-study', 'blog',
                'news', 'career', 'job', 'pricing', 'plan'
            ]

            prioritized_links = []
            for keyword in priority_keywords:
                for link in links:
                    if keyword in link.lower() and link not in prioritized_links:
                        prioritized_links.append(link)

            for link in links:
                if link not in prioritized_links:
                    prioritized_links.append(link)

            urls_to_crawl.extend(prioritized_links[:Config.MAX_PAGES-1])

        except Exception as e:
            print(f"Error discovering pages: {e}")

        return list(set(urls_to_crawl))

    def crawl_domain(self, domain: str, sync_mode=False) -> Dict:
        print(f"Starting to crawl: {domain} (Sync mode: {sync_mode})")

        urls_to_crawl = self.discover_pages(domain)
        crawled_data = []
        updated_pages = []
        new_pages = []

        for url in urls_to_crawl:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    content_data = self.extract_content(response.text, url)

                    if content_data['word_count'] > 50:
                        crawled_data.append(content_data)

                        if sync_mode and url in self.last_crawl_hashes:
                            if self.last_crawl_hashes[url] != content_data['content_hash']:
                                updated_pages.append(url)
                        elif sync_mode:
                            new_pages.append(url)

                        self.last_crawl_hashes[url] = content_data['content_hash']

                self.visited_urls.add(url)
                time.sleep(Config.CRAWL_DELAY)

            except Exception as e:
                print(f"Error crawling {url}: {e}")
                continue

        sync_info = {
            'updated_pages': updated_pages,
            'new_pages': new_pages,
            'total_changes': len(updated_pages) + len(new_pages)
        } if sync_mode else {}

        return {
            'domain': domain,
            'pages': crawled_data,
            'total_pages': len(crawled_data),
            'crawl_date': datetime.now().isoformat(),
            'sync_info': sync_info
        }

# 1. ADD THIS METHOD TO EnhancedDomainCrawler CLASS
    def crawl_specific_urls(self, urls: List[str]) -> Dict:
        """Crawl specific URLs instead of entire domain"""
        print(f"Starting to crawl {len(urls)} specific URLs")
            
        crawled_data = []
        failed_urls = []
            
        for url in urls:
            try:
                # Ensure URL has proper scheme
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                    
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    content_data = self.extract_content(response.text, url)
                        
                    if content_data['word_count'] > 50:
                        crawled_data.append(content_data)
                        print(f"✓ Successfully crawled: {url}")
                    else:
                        print(f"⚠ Skipped (too short): {url}")
                else:
                    failed_urls.append(url)
                    print(f"✗ Failed to crawl: {url} (Status: {response.status_code})")
                        
                time.sleep(Config.CRAWL_DELAY)
                    
            except Exception as e:
                failed_urls.append(url)
                print(f"✗ Error crawling {url}: {e}")
                continue
            
        return {
            'domain': 'multiple-urls',
            'urls': urls,
            'pages': crawled_data,
            'total_pages': len(crawled_data),
            'failed_urls': failed_urls,
            'crawl_date': datetime.now().isoformat(),
            'crawl_type': 'specific_urls'
        }        

# =============================================================================
# ENHANCED CONTENT PROCESSOR CLASS
# =============================================================================
class EnhancedContentProcessor:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.domain_metadata = {}

    def create_chunks(self, content: str, metadata: dict) -> List[Dict]:
        words = content.split()
        chunks = []

        for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
            chunk_words = words[i:i + Config.CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)

            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = len(chunks)
            chunk_metadata['chunk_size'] = len(chunk_words)

            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })

        return chunks

    def process_domain_data(self, domain_data: Dict, sync_mode=False) -> str:
        # ✅ Safely handle both full domains and specific URLs
        domain_key = domain_data.get('domain', 'multiple-urls')
        collection_name = f"domain_{hash(domain_key) % 10000}"

        if not sync_mode:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            self.collection = self.chroma_client.create_collection(collection_name)
        else:
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except:
                self.collection = self.chroma_client.create_collection(collection_name)

        all_chunks = []

        self.domain_metadata = {
            'domain': domain_key,
            'last_crawl': domain_data['crawl_date'],
            'total_pages': domain_data['total_pages']
        }

        for page in domain_data['pages']:
            page_metadata = {
                'url': page['url'],
                'title': page['title'],
                'headings': json.dumps(page['headings']),
                'word_count': page['word_count'],
                'content_hash': page['content_hash'],
                'timestamp': page['timestamp']
            }

            chunks = self.create_chunks(page['content'], page_metadata)
            all_chunks.extend(chunks)

        if all_chunks:
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(texts).tolist()

            if sync_mode:
                start_id = self.collection.count() if hasattr(self.collection, 'count') else 0
                ids = [f"chunk_{start_id + i}" for i in range(len(all_chunks))]
            else:
                ids = [f"chunk_{i}" for i in range(len(all_chunks))]

            metadatas = [chunk['metadata'] for chunk in all_chunks]

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

        mode_text = "Updated" if sync_mode else "Processed"
        result_text = f"{mode_text} {len(all_chunks)} chunks from {len(domain_data['pages'])} pages"

        if sync_mode and 'sync_info' in domain_data:
            sync_info = domain_data['sync_info']
            result_text += f"\nSync Results: {sync_info['total_changes']} changes detected"

        return result_text

    def search_similar_content(self, query: str, n_results: int = Config.CONTEXT_CHUNKS) -> List[Dict]:
        if not self.collection:
            return []

        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        similar_content = []
        for i in range(len(results['documents'][0])):
            similar_content.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })

        return similar_content

# =============================================================================
# ENHANCED GROQ API INTEGRATION
# =============================================================================
class EnhancedGroqAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.chat_history = []

    def add_to_chat_history(self, user_message: str, assistant_response: str):
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })
        self.chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        if len(self.chat_history) > Config.MAX_CHAT_HISTORY * 2:
            self.chat_history = self.chat_history[-(Config.MAX_CHAT_HISTORY * 2):]

    def generate_response_with_context(self, query: str, context_chunks: List[Dict], domain_info: Dict = None) -> str:
        context = ""
        sources = []

        if context_chunks:
            context = "RELEVANT WEBSITE CONTENT:\n"
            for i, chunk in enumerate(context_chunks, 1):
                context += f"\n[Context {i}]\nSource: {chunk['metadata']['url']}\nContent: {chunk['text'][:600]}...\n"
                sources.append(chunk['metadata']['url'])

        domain_context = ""
        if domain_info:
            domain_context = f"\nDOMAIN INFO:\nWebsite: {domain_info.get('domain', 'N/A')}\nLast Updated: {domain_info.get('last_crawl', 'N/A')}\nTotal Pages: {domain_info.get('total_pages', 'N/A')}\n"

        system_message = {
            "role": "system",
            "content": f"""You are an AI assistant specialized in analyzing website content and answering questions about businesses and organizations.

You have access to crawled website content and should provide detailed, accurate answers based on this information. Always cite sources when possible.

{domain_context}

Key Guidelines:
- Answer based on the provided website content
- Be specific and detailed in your responses
- If information isn't available in the content, state that clearly
- Maintain conversation context from previous messages
- Provide helpful insights and analysis
- Include relevant source URLs when citing information
"""
        }

        messages = [system_message]
        messages.extend(self.chat_history)

        user_message = f"{context}\n\nUSER QUESTION: {query}"
        messages.append({
            "role": "user",
            "content": user_message
        })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": Config.GROQ_MODEL,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            assistant_response = response.json()['choices'][0]['message']['content']

            if sources:
                assistant_response += f"\n\n📚 **Sources:**\n"
                for source in list(set(sources)):
                    assistant_response += f"- {source}\n"

            self.add_to_chat_history(query, assistant_response)
            return assistant_response

        except Exception as e:
            error_response = f"Error generating response: {str(e)}"
            self.add_to_chat_history(query, error_response)
            return error_response

    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": Config.GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert business analyst specializing in analyzing companies and websites. Provide detailed, accurate, and well-structured analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def clear_chat_history(self):
        self.chat_history = []

# =============================================================================
# ENHANCED MAIN DOMAIN ANALYZER CLASS
# =============================================================================
class EnhancedDomainAnalyzer:
    def __init__(self, groq_api_key: str = None):
        self.crawler = EnhancedDomainCrawler()
        self.processor = EnhancedContentProcessor()
        self.analyzer = EnhancedGroqAnalyzer(groq_api_key or Config.GROQ_API_KEY)
        self.current_domain_data = None
        self.current_domain = None
        self.last_sync_time = None

    def analyze_domain(self, domain: str) -> Tuple[str, str]:
        if not domain.startswith(('http://', 'https://')):
            domain = 'https://' + domain

        try:
            domain_data = self.crawler.crawl_domain(domain)
            self.current_domain_data = domain_data
            self.current_domain = domain
            self.last_sync_time = datetime.now()

            if not domain_data['pages']:
                return "❌ Error: Could not crawl any pages from the domain", ""

            process_result = self.processor.process_domain_data(domain_data)
            summary = self.generate_domain_summary(domain_data)
            crawl_info = self.generate_crawl_report(domain_data)

            self.analyzer.clear_chat_history()

            return summary, crawl_info

        except Exception as e:
            return f"❌ Error analyzing domain: {str(e)}", ""

            
    # 2. ADD THIS METHOD TO EnhancedDomainAnalyzer CLASS
    def analyze_specific_urls(self, urls: List[str]) -> Tuple[str, str]:
        """Analyze specific URLs instead of entire domain"""
        try:
            # Clean and validate URLs
            cleaned_urls = []
            for url in urls:
                url = url.strip()
                if url:
                    cleaned_urls.append(url)
            
            if not cleaned_urls:
                return "❌ Error: No valid URLs provided", ""
            
            crawl_data = self.crawler.crawl_specific_urls(cleaned_urls)
            self.current_domain_data = crawl_data
            self.current_domain = "multiple_urls"
            self.last_sync_time = datetime.now()
            
            if not crawl_data['pages']:
                failed_info = ""
                if crawl_data['failed_urls']:
                    failed_info = f"\nFailed URLs: {', '.join(crawl_data['failed_urls'])}"
                return f"❌ Error: Could not crawl any pages from the provided URLs{failed_info}", ""
            
            process_result = self.processor.process_domain_data(crawl_data)
            summary = self.generate_urls_summary(crawl_data)
            crawl_info = self.generate_urls_crawl_report(crawl_data)
            
            self.analyzer.clear_chat_history()
            
            return summary, crawl_info
            
        except Exception as e:
            return f"❌ Error analyzing URLs: {str(e)}", ""

    # 3. ADD THESE HELPER METHODS TO EnhancedDomainAnalyzer CLASS
    def generate_urls_summary(self, crawl_data: Dict) -> str:
        """Generate summary for specific URLs analysis"""
        all_content = ""
        page_summaries = []
        
        for page in crawl_data['pages'][:10]:
            page_summary = f"Page: {page['title']}\nURL: {page['url']}\nContent: {page['content'][:500]}...\n\n"
            page_summaries.append(page_summary)
        
        all_content = "\n".join(page_summaries)
        
        prompt = f"""
Analyze the following web pages content and provide a comprehensive analysis:

    WEB PAGES CONTENT:
    {all_content}

    Please provide a detailed analysis covering:

    1. **CONTENT OVERVIEW**
    - What type of content/information is presented across these pages?
    - What are the main topics or themes?
    - What is the purpose of these pages?

    2. **KEY INFORMATION EXTRACTED**
    - Important facts, data, or insights found
    - Products, services, or offerings mentioned
    - Contact information or key details

    3. **CONTENT ANALYSIS**
    - Quality and depth of information
    - Target audience
    - Content structure and organization

    4. **INSIGHTS & PATTERNS**
    - Common themes across pages
    - Unique or notable findings
    - Potential use cases or applications

    5. **SUMMARY**
    - Overall assessment of the content
    - Key takeaways
    - Relevance and value of information

    Format your response in clear sections with bullet points where appropriate. Be specific and cite information found on the pages.
    """
        
        return self.analyzer.generate_response(prompt, max_tokens=1500)

    def generate_urls_crawl_report(self, crawl_data: Dict) -> str:
        """Generate crawl report for specific URLs"""
        pages = crawl_data['pages']
        total_words = sum(page['word_count'] for page in pages)
        avg_words = total_words / len(pages) if pages else 0
        
        report = f"""
    ## 📊 URL Crawl Report

    **URLs Provided:** {len(crawl_data['urls'])}
    **Successfully Crawled:** {len(pages)}
    **Failed URLs:** {len(crawl_data.get('failed_urls', []))}
    **Total Words:** {total_words:,}
    **Average Words per Page:** {avg_words:.0f}
    **Crawl Date:** {crawl_data['crawl_date'][:19]}
    """
        
        if crawl_data.get('failed_urls'):
            report += f"\n### ❌ Failed URLs:\n"
            for url in crawl_data['failed_urls']:
                report += f"- {url}\n"
        
        report += "\n### ✅ Successfully Crawled Pages:\n"
        
        for i, page in enumerate(pages, 1):
            report += f"{i}. **{page['title']}** ({page['word_count']} words)\n   `{page['url']}`\n\n"
        
        return report


    def sync_domain(self) -> str:
        if not self.current_domain:
            return "❌ No domain to sync. Please analyze a domain first."
        
        # Add this check for multiple URLs
        if self.current_domain == "multiple_urls":
            return "❌ Sync is not supported for specific URLs analysis. Please re-analyze the URLs to get updated content."
        
        try:
            domain_data = self.crawler.crawl_domain(self.current_domain, sync_mode=True)
            process_result = self.processor.process_domain_data(domain_data, sync_mode=True)
            self.current_domain_data = domain_data
            self.last_sync_time = datetime.now()
            
            return f"✅ Sync completed!\n\n{process_result}"
            
        except Exception as e:
            return f"❌ Error syncing domain: {str(e)}"

    def chat_with_domain(self, message: str) -> str:
        if not self.processor.collection:
            return "❌ No domain data available. Please analyze a domain first."

        relevant_content = self.processor.search_similar_content(message)

        if not relevant_content:
            return "❌ No relevant information found for your question. Try asking about specific aspects of the website or business."

        domain_info = self.processor.domain_metadata

        response = self.analyzer.generate_response_with_context(
            message,
            relevant_content,
            domain_info
        )

        return response

    def get_chat_history(self) -> List[Dict]:
        return self.analyzer.chat_history

    def clear_chat_history(self) -> str:
        self.analyzer.clear_chat_history()
        return "✅ Chat history cleared!"

    def generate_domain_summary(self, domain_data: Dict) -> str:
        all_content = ""
        page_summaries = []

        for page in domain_data['pages'][:10]:
            page_summary = f"Page: {page['title']}\nURL: {page['url']}\nContent: {page['content'][:500]}...\n\n"
            page_summaries.append(page_summary)

        all_content = "\n".join(page_summaries)

        prompt = f"""
Analyze the following website content and provide a comprehensive business analysis:

WEBSITE CONTENT:
{all_content}

Please provide a detailed analysis covering:

1. **BUSINESS OVERVIEW**
   - What type of business/organization is this?
   - What is their main purpose or mission?
   - What industry do they operate in?

2. **PRODUCTS & SERVICES**
   - What products or services do they offer?
   - Who is their target audience?
   - What makes them unique?

3. **KEY INFORMATION**
   - Contact information
   - Location/markets served
   - Company size or scale

4. **DIGITAL PRESENCE**
   - Website quality and user experience
   - Key features or functionalities
   - Content strategy

5. **BUSINESS INSIGHTS**
   - Market positioning
   - Competitive advantages
   - Recent developments or news

Format your response in clear sections with bullet points where appropriate. Be specific and cite information found on their website.
"""

        return self.analyzer.generate_response(prompt, max_tokens=1500)

    def generate_crawl_report(self, domain_data: Dict) -> str:
        pages = domain_data['pages']
        total_words = sum(page['word_count'] for page in pages)
        avg_words = total_words / len(pages) if pages else 0

        report = f"""
## 📊 Crawl Report

**Domain:** {domain_data['domain']}
**Pages Analyzed:** {len(pages)}
**Total Words:** {total_words:,}
**Average Words per Page:** {avg_words:.0f}
**Crawl Date:** {domain_data['crawl_date'][:19]}
"""

        if 'sync_info' in domain_data and domain_data['sync_info']:
            sync_info = domain_data['sync_info']
            report += f"""
**Sync Status:** ✅ Up to date
**Total Changes:** {sync_info['total_changes']}
**New Pages:** {len(sync_info.get('new_pages', []))}
**Updated Pages:** {len(sync_info.get('updated_pages', []))}
"""

        report += "\n### 📄 Pages Discovered:\n"

        for i, page in enumerate(pages[:15], 1):
            report += f"{i}. **{page['title']}** ({page['word_count']} words)\n   `{page['url']}`\n\n"

        if len(pages) > 15:
            report += f"... and {len(pages) - 15} more pages\n"

        return report

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================
analyzer_instances = {}

# =============================================================================
# FLASK API ROUTES
# =============================================================================


@app.route('/api/initialize', methods=['POST'])
def initialize_analyzer():
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        
        if not api_key or api_key == "your_groq_api_key_here":
            return jsonify({
                'success': False,
                'message': 'Please enter a valid Groq API key. Get one free at https://console.groq.com'
            })
        
        session_id = str(uuid.uuid4())
        analyzer_instances[session_id] = EnhancedDomainAnalyzer(api_key)
        
        return jsonify({
            'success': True,
            'message': 'Analyzer initialized successfully!',
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error initializing analyzer: {str(e)}'
        })

@app.route('/api/analyze', methods=['POST'])
def analyze_domain():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        domain = data.get('domain')
        
        if session_id not in analyzer_instances:
            return jsonify({
                'success': False,
                'message': 'Session not found. Please initialize first.'
            })
        
        analyzer = analyzer_instances[session_id]
        summary, crawl_info = analyzer.analyze_domain(domain)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'crawl_info': crawl_info,
            'message': 'Domain analyzed successfully!'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error analyzing domain: {str(e)}'
        })

# 4. ADD THIS NEW FLASK ROUTE
@app.route('/api/analyze-urls', methods=['POST'])
def analyze_urls():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        urls = data.get('urls', [])
        
        if session_id not in analyzer_instances:
            return jsonify({
                'success': False,
                'message': 'Session not found. Please initialize first.'
            })
        
        if not urls:
            return jsonify({
                'success': False,
                'message': 'No URLs provided. Please enter at least one URL.'
            })
        
        analyzer = analyzer_instances[session_id]
        summary, crawl_info = analyzer.analyze_specific_urls(urls)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'crawl_info': crawl_info,
            'message': 'URLs analyzed successfully!'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error analyzing URLs: {str(e)}'
        })
    
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message')
        
        if session_id not in analyzer_instances:
            return jsonify({
                'success': False,
                'message': 'Session not found. Please initialize first.'
            })
        
        analyzer = analyzer_instances[session_id]
        response = analyzer.chat_with_domain(message)
        
        return jsonify({
            'success': True,
            'response': response,
            'chat_history': analyzer.get_chat_history()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error in chat: {str(e)}'
        })

@app.route('/api/sync', methods=['POST'])
def sync_domain():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in analyzer_instances:
            return jsonify({
                'success': False,
                'message': 'Session not found. Please initialize first.'
            })
        
        analyzer = analyzer_instances[session_id]
        result = analyzer.sync_domain()
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error syncing domain: {str(e)}'
        })

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in analyzer_instances:
            return jsonify({
                'success': False,
                'message': 'Session not found. Please initialize first.'
            })
        
        analyzer = analyzer_instances[session_id]
        result = analyzer.clear_chat_history()
        
        return jsonify({
            'success': True,
            'message': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error clearing chat: {str(e)}'
        })

@app.route('/api/upload-html', methods=['POST'])
def upload_html():
    try:
        session_id = request.form.get('session_id')
        if session_id not in analyzer_instances:
            return jsonify({'success': False, 'message': 'Session not found. Please initialize first.'})

        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No HTML file uploaded'})

        file = request.files['file']
        if not file.filename.endswith('.html'):
            return jsonify({'success': False, 'message': 'Only .html files are supported'})

        html_content = file.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')

        # Clean & extract text
        for tag in soup(['script', 'style']):
            tag.decompose()
        extracted_text = soup.get_text(separator=' ', strip=True)

        if not extracted_text or len(extracted_text.split()) < 30:
            return jsonify({'success': False, 'message': 'HTML file has too little content to analyze'})

        # Generate summary
        analyzer = analyzer_instances[session_id]
        prompt = f"""
Analyze the following HTML document content and summarize its purpose, content, and key takeaways.

CONTENT:
{extracted_text[:5000]}  # truncate to avoid overload

Please provide:
1. Summary
2. Purpose
3. Key details or findings
"""

        summary = analyzer.analyzer.generate_response(prompt, max_tokens=1200)

        return jsonify({'success': True, 'summary': summary})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing HTML: {str(e)}'})
@app.route('/')
def index():
    return render_template('ui2.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'success': True,
        'active_sessions': len(analyzer_instances),
        'message': 'API is running'
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)

    print("🌐 Enhanced Domain Intelligence Analyzer - Flask API")
    print("=" * 50)
    print("🚀 Starting Flask API server...")
    print("📱 Access the web interface at: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
