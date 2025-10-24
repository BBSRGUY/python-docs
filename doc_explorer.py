#!/usr/bin/env python3
"""
🐍 Python Docs Explorer - An Interactive Documentation Tool
Surprise features included!
"""

import os
import re
import random
from pathlib import Path
from collections import defaultdict


class PythonDocsExplorer:
    def __init__(self):
        self.docs_dir = Path(__file__).parent
        self.docs = self._load_all_docs()
        self.code_snippets = self._extract_code_snippets()

    def _load_all_docs(self):
        """Load all markdown documentation files"""
        docs = {}
        for md_file in self.docs_dir.glob("python-*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                docs[md_file.stem] = f.read()
        return docs

    def _extract_code_snippets(self):
        """Extract all Python code snippets from documentation"""
        snippets = defaultdict(list)
        code_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)

        for doc_name, content in self.docs.items():
            matches = code_pattern.findall(content)
            snippets[doc_name] = [m.strip() for m in matches if m.strip()]

        return snippets

    def search(self, query):
        """Search across all documentation"""
        results = []
        query_lower = query.lower()

        for doc_name, content in self.docs.items():
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    results.append({
                        'doc': doc_name,
                        'line': i + 1,
                        'content': line.strip()
                    })

        return results

    def random_snippet(self, topic=None):
        """Get a random code snippet, optionally filtered by topic"""
        if topic and topic in self.code_snippets:
            snippets = self.code_snippets[topic]
            source = topic
        else:
            all_snippets = []
            sources = []
            for doc, snips in self.code_snippets.items():
                all_snippets.extend(snips)
                sources.extend([doc] * len(snips))

            if not all_snippets:
                return None, None

            idx = random.randint(0, len(all_snippets) - 1)
            return all_snippets[idx], sources[idx]

        if snippets:
            return random.choice(snippets), source
        return None, None

    def get_stats(self):
        """Get interesting statistics about the documentation"""
        total_lines = sum(len(content.split('\n')) for content in self.docs.values())
        total_code_snippets = sum(len(snippets) for snippets in self.code_snippets.values())
        total_size = sum(len(content) for content in self.docs.values())

        # Count common keywords
        all_text = ' '.join(self.docs.values()).lower()
        keywords = {
            'function': len(re.findall(r'\bfunction\b', all_text)),
            'class': len(re.findall(r'\bclass\b', all_text)),
            'import': len(re.findall(r'\bimport\b', all_text)),
            'def': len(re.findall(r'\bdef\b', all_text)),
            'return': len(re.findall(r'\breturn\b', all_text)),
        }

        return {
            'total_docs': len(self.docs),
            'total_lines': total_lines,
            'total_snippets': total_code_snippets,
            'total_chars': total_size,
            'keywords': keywords
        }

    def python_fact_of_the_day(self):
        """Generate a fun Python fact based on the documentation"""
        facts = [
            f"📚 This documentation contains {self.get_stats()['total_snippets']} code examples!",
            f"📝 There are {self.get_stats()['total_lines']:,} lines of Python wisdom here!",
            f"🔍 The word 'function' appears {self.get_stats()['keywords']['function']} times in these docs!",
            "🐍 Python was named after Monty Python, not the snake!",
            "✨ Python uses indentation for code blocks, making it beautifully readable!",
            f"📖 You have {len(self.docs)} different Python topics to explore!",
            "🚀 Python's philosophy: 'There should be one-- and preferably only one --obvious way to do it.'",
            f"💻 These docs contain {self.get_stats()['total_chars']:,} characters of Python knowledge!",
        ]
        return random.choice(facts)

    def interactive_menu(self):
        """Run an interactive documentation explorer"""
        while True:
            print("\n" + "="*60)
            print("🐍 PYTHON DOCS EXPLORER 🐍".center(60))
            print("="*60)
            print("\n1. 🔍 Search documentation")
            print("2. 🎲 Random code snippet")
            print("3. 📊 Documentation statistics")
            print("4. 💡 Python fact of the day")
            print("5. 📚 List all topics")
            print("6. 🎯 Quiz me! (Random snippet guess)")
            print("7. 🚪 Exit")
            print("\n" + "-"*60)

            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == '1':
                query = input("\n🔍 Enter search term: ").strip()
                results = self.search(query)
                print(f"\n✨ Found {len(results)} results for '{query}':")
                for r in results[:10]:  # Show first 10
                    print(f"  📄 {r['doc']}:{r['line']} - {r['content'][:80]}...")
                if len(results) > 10:
                    print(f"\n  ... and {len(results) - 10} more results")

            elif choice == '2':
                snippet, source = self.random_snippet()
                if snippet:
                    print(f"\n🎲 Random snippet from {source}:")
                    print("\n" + "─"*60)
                    print(snippet)
                    print("─"*60)
                else:
                    print("\n❌ No snippets found!")

            elif choice == '3':
                stats = self.get_stats()
                print("\n📊 Documentation Statistics:")
                print(f"  📚 Total documents: {stats['total_docs']}")
                print(f"  📝 Total lines: {stats['total_lines']:,}")
                print(f"  💻 Code snippets: {stats['total_snippets']}")
                print(f"  📖 Total characters: {stats['total_chars']:,}")
                print(f"\n  🔑 Keyword frequency:")
                for kw, count in stats['keywords'].items():
                    print(f"     {kw}: {count}")

            elif choice == '4':
                print(f"\n💡 {self.python_fact_of_the_day()}")

            elif choice == '5':
                print("\n📚 Available topics:")
                for i, doc_name in enumerate(sorted(self.docs.keys()), 1):
                    topic = doc_name.replace('python-', '').title()
                    snippets_count = len(self.code_snippets.get(doc_name, []))
                    print(f"  {i}. {topic} ({snippets_count} code examples)")

            elif choice == '6':
                snippet, source = self.random_snippet()
                if snippet:
                    print("\n🎯 QUIZ TIME! What topic is this code from?")
                    print("\n" + "─"*60)
                    # Show just first few lines
                    lines = snippet.split('\n')[:5]
                    print('\n'.join(lines))
                    if len(snippet.split('\n')) > 5:
                        print("...")
                    print("─"*60)

                    input("\n🤔 Take a guess, then press Enter to reveal...")
                    topic = source.replace('python-', '').title()
                    print(f"\n✨ Answer: {topic}!")
                else:
                    print("\n❌ No snippets available for quiz!")

            elif choice == '7':
                print("\n👋 Happy coding! May your bugs be few and your code be Pythonic! 🐍")
                break

            else:
                print("\n❌ Invalid choice! Please enter 1-7.")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        🎉 SURPRISE! Python Docs Explorer 🎉              ║
    ║                                                          ║
    ║   An interactive tool to explore your Python docs!       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    explorer = PythonDocsExplorer()

    # Show a welcome fact
    print(f"\n💡 Welcome fact: {explorer.python_fact_of_the_day()}\n")

    try:
        explorer.interactive_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted! Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
