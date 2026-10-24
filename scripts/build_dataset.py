"""Build data/queries.jsonl, the query log used to evaluate the cache and router.

Each line is one prompt:
    id          "q0001", ...
    prompt      the user message
    category    greeting, fact, definition, howto, math, translation, advice,
                code, debugging, reasoning, writing, summarization, explanation
    difficulty  "easy" or "hard" (router labels)
    dup_of      id of the original prompt if this is a near-duplicate: same meaning,
                different wording. The semantic cache SHOULD hit.
    trap_of     id of a look-alike prompt with a different answer.
                The semantic cache must NOT hit.

Run: uv run python scripts/build_dataset.py
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "queries.jsonl"


@dataclass
class Seed:
    category: str
    difficulty: str
    prompt: str
    dups: list[str] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)


def easy(category, prompt, dups=(), traps=()):
    return Seed(category, "easy", prompt, list(dups), list(traps))


def hard(category, prompt, dups=(), traps=()):
    return Seed(category, "hard", prompt, list(dups), list(traps))


REMOTE_WORK_TEXT = (
    "Remote work grew rapidly after 2020. Companies found that many roles could be done "
    "from home without a drop in output, and employees valued the time saved on commuting. "
    "However, managers reported that onboarding new hires and building team culture became "
    "harder, and some employees felt isolated. Many firms have now settled on hybrid models, "
    "asking staff to come into the office two or three days a week, while investing in "
    "better collaboration tools and clearer written communication."
)

SEEDS: list[Seed] = [
    # --- greeting / chit-chat ---
    easy("greeting", "Hi there!", dups=["hi there", "Hello there!"]),
    easy("greeting", "Good morning! How are you today?", dups=["good morning, how are you?"]),
    easy("greeting", "Thanks, that was really helpful."),
    easy("greeting", "Can you tell me a joke?", dups=["Tell me a joke."]),
    easy("greeting", "What can you help me with?", dups=["what can you do?"]),
    easy("greeting", "Tell me a fun fact about space."),
    easy("greeting", "Recommend a good book for a long train journey."),
    easy("greeting", "What's a good name for a pet goldfish?"),
    easy("greeting", "Say something motivational for a Monday morning."),
    # --- facts ---
    easy(
        "fact",
        "What is the capital of France?",
        dups=["capital of france?", "Which city is the capital of France?"],
        traps=["What is the capital of Germany?"],
    ),
    easy(
        "fact",
        "Who wrote Pride and Prejudice?",
        dups=["who is the author of pride and prejudice"],
        traps=["Who wrote Pride and Prejudice and Zombies?"],
    ),
    easy(
        "fact",
        "How many planets are in the solar system?",
        dups=["how many planets are there in our solar system"],
    ),
    easy(
        "fact",
        "What is the boiling point of water in Celsius?",
        dups=["boiling point of water in celsius"],
        traps=["What is the boiling point of water in Fahrenheit?"],
    ),
    easy(
        "fact",
        "What is the largest ocean on Earth?",
        traps=["What is the smallest ocean on Earth?"],
    ),
    easy("fact", "Who painted the Mona Lisa?", dups=["who painted mona lisa"]),
    easy(
        "fact",
        "What year did India gain independence?",
        dups=["When did India become independent?"],
        traps=["What year did India become a republic?"],
    ),
    easy(
        "fact",
        "What is the chemical symbol for gold?",
        dups=["chemical symbol of gold?"],
        traps=["What is the chemical symbol for silver?"],
    ),
    easy(
        "fact",
        "How many bones are in the adult human body?",
        dups=["how many bones does an adult human have"],
    ),
    easy(
        "fact",
        "What is the tallest mountain in the world?",
        dups=["Which is the highest mountain on Earth?"],
        traps=["What is the tallest mountain in India?"],
    ),
    easy(
        "fact",
        "Who was the first person to walk on the Moon?",
        dups=["who first walked on the moon"],
        traps=["Who was the second person to walk on the Moon?"],
    ),
    easy("fact", "What is the speed of light in a vacuum?", dups=["speed of light in vacuum"]),
    easy(
        "fact",
        "What is the currency of Japan?",
        dups=["what currency does japan use"],
        traps=["What is the currency of China?"],
    ),
    easy("fact", "What is the longest river in the world?"),
    easy("fact", "How many continents are there?", dups=["how many continents are there on earth"]),
    easy(
        "fact",
        "What is the national animal of India?",
        traps=["What is the national bird of India?"],
    ),
    easy("fact", "What does DNA stand for?", dups=["DNA stands for what?"]),
    easy("fact", "What does CPU stand for?", traps=["What does GPU stand for?"]),
    easy("fact", "Who invented the telephone?"),
    easy("fact", "What is the freezing point of water in Fahrenheit?"),
    easy("fact", "How many days are in a leap year?", dups=["how many days does a leap year have"]),
    easy(
        "fact",
        "What is the largest planet in our solar system?",
        dups=["Which planet is the biggest in the solar system?"],
        traps=["What is the smallest planet in our solar system?"],
    ),
    easy("fact", "Who is known as the father of computers?"),
    easy(
        "fact",
        "What language is spoken in Brazil?",
        dups=["what language do people speak in brazil"],
        traps=["What language is spoken in Argentina?"],
    ),
    easy("fact", "What is the hardest natural substance?"),
    easy(
        "fact",
        "In which year did World War II end?",
        dups=["when did world war 2 end"],
        traps=["In which year did World War I end?"],
    ),
    easy("fact", "What is the main gas in Earth's atmosphere?"),
    easy(
        "fact",
        "How many players are on a cricket team?",
        traps=["How many players are on a football team?"],
    ),
    easy("fact", "What is the largest desert in the world?"),
    easy(
        "fact",
        "Who developed the theory of relativity?",
        dups=["who came up with the theory of relativity"],
    ),
    easy(
        "fact",
        "What is the capital of Australia?",
        dups=["capital city of australia"],
        traps=["What is the largest city in Australia?"],
    ),
    easy("fact", "What time zone is India in?", dups=["what timezone is india in"]),
    easy("fact", "Roughly what is the population of India?"),
    easy("fact", "Which organ pumps blood through the body?"),
    easy(
        "fact",
        "What is the default port for HTTPS?",
        dups=["default https port?"],
        traps=["What is the default port for HTTP?"],
    ),
    easy(
        "fact",
        "What is the default port for PostgreSQL?",
        dups=["which port does postgres use by default"],
        traps=["What is the default port for MySQL?"],
    ),
    easy("fact", "Who created Python?", dups=["who invented the python programming language"]),
    easy("fact", "What year was Python 3 released?"),
    easy("fact", "What does HTTP stand for?"),
    easy("fact", "What does JSON stand for?", dups=["json stands for?"]),
    # Answer changes over time: a good example of why cache entries need a TTL.
    easy("fact", "What is the latest stable version of Python?"),
    easy("fact", "How many bits are in a byte?", dups=["how many bits in one byte"]),
    easy(
        "fact",
        "What is the binary representation of 10?",
        traps=["What is the binary representation of 12?"],
    ),
    # --- definitions ---
    easy(
        "definition",
        "What is machine learning?",
        dups=["what is ML?", "Explain machine learning in simple terms."],
    ),
    easy("definition", "What is an API?", dups=["what is an api"]),
    easy(
        "definition",
        "What is the difference between RAM and ROM?",
        traps=["What is the difference between RAM and an SSD?"],
    ),
    easy("definition", "What is a REST API?"),
    easy("definition", "Define photosynthesis.", dups=["what is photosynthesis"]),
    easy("definition", "What is inflation?", dups=["What does inflation mean in economics?"]),
    easy("definition", "What is a black hole?"),
    easy("definition", "What is the cloud in computing?"),
    easy("definition", "What is a large language model?", dups=["what is an LLM?"]),
    easy(
        "definition",
        "What is an embedding in machine learning?",
        dups=["what are embeddings in ML"],
    ),
    easy("definition", "What is a vector database?"),
    easy("definition", "What is Docker?", dups=["what is docker used for"]),
    easy("definition", "What is Kubernetes, in one sentence?"),
    easy("definition", "What is Git?"),
    easy("definition", "What is the difference between Git and GitHub?", dups=["git vs github"]),
    easy("definition", "What is a prompt injection attack?"),
    easy(
        "definition",
        "What is overfitting in machine learning?",
        dups=["what does overfitting mean"],
        traps=["What is underfitting in machine learning?"],
    ),
    easy("definition", "What is a neural network?"),
    easy("definition", "What is SQL?"),
    easy("definition", "What is the difference between TCP and UDP?", dups=["tcp vs udp"]),
    easy("definition", "What is an IP address?"),
    easy("definition", "What is two-factor authentication?", dups=["what is 2FA"]),
    easy("definition", "What is a mutual fund?"),
    easy("definition", "What is GDP?", dups=["what does GDP mean"]),
    easy("definition", "What is a compiler?", traps=["What is a cross-compiler?"]),
    easy("definition", "What is recursion?", dups=["explain recursion"]),
    easy("definition", "What is a hash function?"),
    easy("definition", "What is HTTPS and why is it secure?"),
    easy(
        "definition",
        "What is the difference between a list and a tuple in Python?",
        dups=["python list vs tuple"],
        traps=["What is the difference between a list and a set in Python?"],
    ),
    easy("definition", "What is a virtual environment in Python?"),
    easy("definition", "What is async/await in Python?"),
    easy("definition", "What is a token in the context of LLMs?", dups=["what is a token in LLMs"]),
    easy(
        "definition",
        "What does temperature do in an LLM API?",
        dups=["what is the temperature parameter in llms"],
        traps=["What does top_p do in an LLM API?"],
    ),
    easy("definition", "What is retrieval-augmented generation?", dups=["what is RAG in AI"]),
    easy("definition", "What is a CDN?"),
    easy(
        "definition",
        "Why does 0.1 + 0.2 not equal 0.3 in Python?",
        dups=["why is 0.1+0.2 != 0.3 in python"],
    ),
    # --- how-to ---
    easy(
        "howto",
        "How do I reverse a string in Python?",
        dups=["reverse a string in python", "How can I reverse a string using Python?"],
        traps=["How do I reverse a list in Python?"],
    ),
    easy(
        "howto",
        "How do I create a virtual environment in Python?",
        dups=["how to make a python venv"],
    ),
    easy("howto", "How do I install a package with pip?"),
    easy(
        "howto",
        "How do I undo the last git commit?",
        dups=["how to undo last commit in git"],
        traps=["How do I undo the last git push?"],
    ),
    easy(
        "howto", "How do I check which Python version I have?", dups=["how to check python version"]
    ),
    easy(
        "howto",
        "How do I list all running Docker containers?",
        dups=["docker command to list running containers"],
        traps=["How do I list all Docker images?"],
    ),
    easy("howto", "How do I read a CSV file in pandas?", dups=["read csv with pandas"]),
    easy(
        "howto",
        "How do I sort a dictionary by value in Python?",
        dups=["python sort dict by value"],
        traps=["How do I sort a dictionary by key in Python?"],
    ),
    easy("howto", "How do I find the length of a list in Python?"),
    easy("howto", "How do I exit Vim?", dups=["how to quit vim"]),
    easy(
        "howto",
        "How do I convert a string to an integer in JavaScript?",
        traps=["How do I convert an integer to a string in JavaScript?"],
    ),
    easy(
        "howto", "How do I make a GET request in Python?", dups=["python http get request example"]
    ),
    easy(
        "howto",
        "How do I boil an egg?",
        dups=["how to boil eggs"],
        traps=["How do I poach an egg?"],
    ),
    easy("howto", "How do I make a cup of masala chai?"),
    easy("howto", "How long should I cook rice in a pressure cooker?"),
    easy(
        "howto",
        "How do I take a screenshot on Windows?",
        dups=["how to screenshot on windows"],
        traps=["How do I take a screenshot on a Mac?"],
    ),
    easy(
        "howto",
        "How do I center a div in CSS?",
        dups=["css center a div", "How can I center a div horizontally and vertically?"],
    ),
    easy(
        "howto",
        "How do I remove duplicates from a list in Python?",
        dups=["python remove duplicates from list"],
    ),
    easy("howto", "How do I check disk usage in Linux?", dups=["linux command for disk usage"]),
    easy("howto", "How do I find a file by name in Linux?"),
    easy("howto", "How do I rename a git branch?"),
    easy("howto", "How do I merge two dictionaries in Python?"),
    easy(
        "howto",
        "How do I get the current date in Python?",
        traps=["How do I get the current time zone in Python?"],
    ),
    easy("howto", "How do I write a for loop in Bash?"),
    easy(
        "howto", "How do I kill a process running on port 8000?", dups=["kill process on port 8000"]
    ),
    # --- easy math ---
    easy("math", "What is 15% of 200?", dups=["15 percent of 200"], traps=["What is 25% of 200?"]),
    easy("math", "What is 12 times 13?"),
    easy(
        "math",
        "Convert 100 kilometres to miles.",
        dups=["100 km in miles"],
        traps=["Convert 100 miles to kilometres."],
    ),
    easy("math", "Convert 30 degrees Celsius to Fahrenheit."),
    easy("math", "What is the area of a circle with radius 7?"),
    easy("math", "Is 97 a prime number?", traps=["Is 91 a prime number?"]),
    easy("math", "What is 2 to the power of 10?", dups=["2^10"]),
    easy("math", "How many seconds are in a day?"),
    easy("math", "What is the average of 4, 8, 15, 16, 23 and 42?"),
    easy("math", "Simplify the fraction 48/64."),
    easy(
        "math",
        "What is the square root of 144?",
        dups=["sqrt of 144"],
        traps=["What is the square root of 169?"],
    ),
    # --- translation ---
    easy(
        "translation",
        "Translate 'good morning' into Hindi.",
        dups=["good morning in hindi"],
        traps=["Translate 'good night' into Hindi."],
    ),
    easy("translation", "Translate 'where is the train station?' into French."),
    easy("translation", "How do you say 'thank you' in Tamil?"),
    easy("translation", "How do I say thank you in Japanese?"),
    easy("translation", "Translate 'I love programming' into Spanish."),
    easy("translation", "What does 'carpe diem' mean?"),
    # --- advice ---
    easy("advice", "How can I sleep better at night?", dups=["tips to sleep better"]),
    easy(
        "advice",
        "Give me three tips to stay focused while studying.",
        dups=["how to stay focused while studying"],
    ),
    easy("advice", "What should I pack for a weekend trip to the mountains?"),
    easy("advice", "How much water should I drink in a day?"),
    easy("advice", "Suggest a healthy breakfast that takes under 10 minutes."),
    easy("advice", "How do I prepare for a job interview?", dups=["tips for a job interview"]),
    # --- code ---
    hard(
        "code",
        "Write a Python function that checks whether a string is a valid palindrome, "
        "ignoring case and non-alphanumeric characters, with unit tests.",
        dups=[
            "Write a python function to check if a string is a palindrome ignoring case "
            "and punctuation, and include unit tests."
        ],
    ),
    hard(
        "code",
        "Implement an LRU cache in Python without using functools, with O(1) get and put.",
        dups=["implement an LRU cache in python with O(1) get and put, no functools"],
    ),
    hard(
        "code",
        "Write a FastAPI endpoint that accepts a file upload, validates it is a CSV "
        "under 5 MB, and returns the row count.",
    ),
    hard(
        "code",
        "Write a SQL query to find the second highest salary from an employees table.",
        dups=["sql query for second highest salary in employees table"],
        traps=["Write a SQL query to find the third highest salary from an employees table."],
    ),
    hard(
        "code",
        "Implement a token bucket rate limiter in Python that is safe to use from asyncio code.",
    ),
    hard(
        "code",
        "Write a Python script that reads a large log file line by line and counts "
        "the top 10 most frequent IP addresses.",
    ),
    hard(
        "code",
        "Write a React component for a todo list with add, delete and mark-as-done, using hooks.",
    ),
    hard(
        "code",
        "Implement binary search in Python and explain its time complexity.",
        dups=["binary search implementation in python with time complexity"],
    ),
    hard("code", "Write a function to merge two sorted linked lists in Java."),
    hard(
        "code",
        "Write a Bash script that backs up a directory to a timestamped tar.gz file "
        "and deletes backups older than 7 days.",
    ),
    hard("code", "Implement Dijkstra's shortest path algorithm in Python using a heap."),
    hard(
        "code",
        "Write a Python decorator that retries a function up to 3 times with "
        "exponential backoff on exceptions.",
        dups=["python retry decorator with exponential backoff, max 3 attempts"],
    ),
    hard(
        "code",
        "Write a regex that validates an Indian mobile number with an optional +91 "
        "prefix, and explain each part.",
    ),
    hard(
        "code",
        "Write a Python async function that fetches 50 URLs concurrently, at most 10 "
        "at a time, using httpx.",
    ),
    hard("code", "Write a Dockerfile for a FastAPI app that uses uv and runs as a non-root user."),
    hard("code", "Implement a trie in Python with insert, search and starts_with methods."),
    hard(
        "code",
        "Write a SQL query that returns each customer's total spend per month, "
        "including months with zero spend.",
    ),
    hard(
        "code",
        "Write a Python class for a bank account with deposit, withdraw and a "
        "transaction history, and handle overdraft errors.",
    ),
    hard(
        "code",
        "Write a pytest fixture that starts a temporary Postgres database for integration tests.",
    ),
    hard(
        "code",
        "Convert this Python 2 code to Python 3:\n\nprint 'hello'\nd = {}\n"
        "if d.has_key('a'):\n    print d['a']",
    ),
    hard("code", "Write a FastAPI endpoint that streams a server-sent events response."),
    hard(
        "code",
        "Write a Python function that splits a long text into chunks of about 500 "
        "tokens with 50 tokens of overlap.",
    ),
    hard("code", "Write a Go HTTP server with a /health endpoint and graceful shutdown."),
    hard(
        "code",
        "Implement quicksort in Python and explain why its worst case is O(n^2).",
        traps=["Implement mergesort in Python and explain why it is always O(n log n)."],
    ),
    hard(
        "code",
        "Write a Python script to compute cosine similarity between two lists of "
        "sentence embeddings using numpy.",
        dups=["numpy code for cosine similarity between two sets of embeddings"],
    ),
    # --- debugging ---
    hard(
        "debugging",
        "Why does this Python code print 4 five times?\n\n"
        "funcs = [lambda: i for i in range(5)]\nfor f in funcs:\n    print(f())",
        dups=[
            "why does this print 4 five times in python?\n\n"
            "funcs = [lambda: i for i in range(5)]\nfor f in funcs:\n    print(f())"
        ],
    ),
    hard(
        "debugging",
        "I get 'RuntimeError: Event loop is closed' when running my asyncio "
        "tests with pytest. What causes this and how do I fix it?",
    ),
    hard(
        "debugging",
        "My Docker container exits immediately after starting with exit code 0. How do I debug it?",
        dups=["docker container exits right away with code 0, how to debug?"],
    ),
    hard(
        "debugging",
        "Why is my SQL query with a LIKE '%term%' filter slow on a table with 10 "
        "million rows, and how can I speed it up?",
    ),
    hard(
        "debugging",
        "My React component re-renders infinitely after I added a useEffect "
        "that sets state. What is wrong?",
    ),
    hard(
        "debugging",
        "I get 'CORS policy: No Access-Control-Allow-Origin header' when calling "
        "my FastAPI backend from React. How do I fix it?",
        dups=[
            "fastapi cors error from react frontend, no access-control-allow-origin "
            "header, how to fix?"
        ],
    ),
    hard(
        "debugging",
        "git says 'refusing to merge unrelated histories'. What does that mean "
        "and how should I resolve it safely?",
    ),
    hard(
        "debugging",
        "My Python script uses 100% of one CPU core but threading doesn't speed "
        "it up. Why, and what should I use instead?",
    ),
    hard(
        "debugging",
        "Postgres says 'too many connections' under load from my FastAPI app. "
        "What is happening and how do I fix it?",
    ),
    hard(
        "debugging",
        "This function is supposed to return the maximum, but gives wrong "
        "answers. Find the bug:\n\ndef find_max(nums):\n    best = None\n    for n in nums:\n"
        "        if best is None or n > best:\n            best = n\n        return best",
    ),
    hard("debugging", "My pandas merge is producing many more rows than either input table. Why?"),
    # --- reasoning / design ---
    hard(
        "reasoning",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
        "the ball. How much does the ball cost? Explain your reasoning.",
        dups=[
            "bat and ball cost 1.10 together, the bat costs 1.00 more than the ball, how "
            "much is the ball? explain"
        ],
        traps=[
            "A bat and a ball cost $1.20 in total. The bat costs $1.00 more than the "
            "ball. How much does the ball cost? Explain your reasoning."
        ],
    ),
    hard(
        "reasoning",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it "
        "take 100 machines to make 100 widgets?",
        traps=[
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take "
            "1 machine to make 100 widgets?"
        ],
    ),
    hard(
        "reasoning",
        "You have 8 balls, one of which is slightly heavier. Using a balance "
        "scale only twice, how do you find the heavier ball?",
    ),
    hard(
        "reasoning",
        "Three friends split a restaurant bill of ₹2,850 including a 10% service "
        "charge. One friend didn't eat the ₹300 dessert. How should they split it fairly?",
    ),
    hard(
        "reasoning",
        "A train leaves Delhi at 6 AM at 80 km/h and another leaves Agra at 7 AM "
        "at 100 km/h towards Delhi. The cities are 230 km apart. When and where do they "
        "meet?",
    ),
    hard("reasoning", "Is it ever rational to buy a lottery ticket? Argue both sides."),
    hard(
        "reasoning",
        "Alice is older than Bob, Carol is younger than Bob, and Dave is older "
        "than Alice. Who is the second oldest? Explain.",
    ),
    hard(
        "reasoning",
        "Compare the pros and cons of microservices versus a monolith for a "
        "3-person startup, and give a recommendation.",
    ),
    hard(
        "reasoning",
        "Should a small team choose PostgreSQL or MongoDB for an e-commerce app? "
        "Justify with trade-offs.",
        dups=["postgres vs mongodb for an e-commerce app built by a small team, which and why?"],
    ),
    hard(
        "reasoning",
        "Design a URL shortener: describe the API, data model, how you generate "
        "short codes, and how it scales to 1,000 requests per second.",
    ),
    hard(
        "reasoning",
        "Design a rate limiter for an API gateway. Compare fixed window, sliding "
        "window and token bucket approaches.",
    ),
    hard(
        "reasoning",
        "Explain how you would design a semantic cache for LLM responses, "
        "including how to avoid returning wrong answers.",
    ),
    hard(
        "reasoning",
        "What are the trade-offs between fine-tuning an LLM and using "
        "retrieval-augmented generation for a company knowledge base?",
    ),
    hard(
        "reasoning",
        "If a model has 95% accuracy on a dataset where 95% of examples are "
        "negative, is it a good model? Explain.",
    ),
    hard(
        "reasoning",
        "Prove that the square root of 2 is irrational.",
        traps=["Prove that the cube root of 2 is irrational."],
    ),
    hard("reasoning", "Prove that there are infinitely many prime numbers."),
    hard(
        "reasoning",
        "Explain the Monty Hall problem and why switching doors is better.",
        dups=["why is switching better in the monty hall problem"],
    ),
    hard(
        "reasoning", "Estimate how many piano tuners there are in Mumbai, showing each assumption."
    ),
    hard("reasoning", "Explain why building a heap takes O(n) time and not O(n log n)."),
    hard(
        "reasoning",
        "How would you detect whether a user's prompt is a jailbreak attempt? "
        "Propose a layered approach.",
    ),
    # --- hard math ---
    hard(
        "math",
        "Solve the system of equations: 3x + 2y = 16 and 5x - y = 18.",
        dups=["solve 3x+2y=16 and 5x-y=18"],
        traps=["Solve the system of equations: 3x + 2y = 16 and 5x + y = 18."],
    ),
    hard("math", "Find the derivative of f(x) = x^3 * sin(x) and explain each step."),
    hard("math", "Integrate x * e^x dx, showing the steps."),
    hard(
        "math",
        "What is the probability of getting at least one six when rolling four dice?",
        traps=["What is the probability of getting exactly one six when rolling four dice?"],
    ),
    hard(
        "math",
        "A loan of ₹5,00,000 at 9% annual interest is repaid monthly over 5 years. "
        "What is the EMI? Show the formula.",
    ),
    hard("math", "Explain eigenvalues and eigenvectors with a 2x2 worked example."),
    hard("math", "How many different ways can 8 people sit around a round table?"),
    hard(
        "math",
        "Calculate the compound interest on ₹1,00,000 at 7% per year compounded "
        "quarterly for 3 years.",
    ),
    # --- writing ---
    hard(
        "writing",
        "Write a cover letter for an entry-level AI engineer role at a fintech "
        "startup, highlighting a project that built an LLM gateway.",
        dups=[
            "write a cover letter for an entry level AI engineer job at a fintech "
            "startup, mention my LLM gateway project"
        ],
    ),
    hard(
        "writing",
        "Write a 300-word blog post introduction about why companies need an LLM gateway.",
    ),
    hard(
        "writing",
        "Write a polite email to my professor asking for a three-day extension on "
        "my assignment deadline.",
        dups=["draft a polite email to my professor asking for a 3-day extension on an assignment"],
        traps=["Write a polite email to my manager asking for three days of leave."],
    ),
    hard(
        "writing", "Write a short story (about 400 words) about a robot who learns to cook biryani."
    ),
    hard(
        "writing", "Write a product description for noise-cancelling headphones aimed at students."
    ),
    hard(
        "writing",
        "Write a LinkedIn post announcing that I completed my first open-source contribution.",
    ),
    hard(
        "writing",
        "Rewrite this paragraph to be more concise and professional: 'So basically "
        "we kind of had a lot of issues with the deployment and stuff, and it took like "
        "forever to fix, but now it is mostly working I think.'",
    ),
    hard(
        "writing",
        "Write a haiku about monsoon rain in Mumbai.",
        traps=["Write a limerick about monsoon rain in Mumbai."],
    ),
    hard(
        "writing",
        "Write a README introduction for an open-source Python library that "
        "validates Indian GST numbers.",
    ),
    hard(
        "writing",
        "Write a project proposal outline for a college hackathon app that helps "
        "students find study partners.",
    ),
    hard("writing", "Draft a polite resignation letter that gives four weeks' notice."),
    hard("writing", "Write a farewell speech of about 200 words for my college batch."),
    hard("writing", "Write an apology message to a customer whose order arrived two weeks late."),
    # --- summarization / extraction ---
    hard(
        "summarization",
        f"Summarise the following in 3 bullet points:\n\n{REMOTE_WORK_TEXT}",
        dups=[f"summarize this in three bullets:\n\n{REMOTE_WORK_TEXT}"],
        traps=[f"Translate the following into Hindi:\n\n{REMOTE_WORK_TEXT}"],
    ),
    hard(
        "summarization",
        "Summarise the key ideas of the book 'Atomic Habits' in five bullet points.",
        dups=["summarize atomic habits in 5 bullets"],
    ),
    hard("summarization", "Summarise the causes of the 2008 financial crisis for a beginner."),
    hard("summarization", "Explain the plot of the Mahabharata in about 200 words."),
    hard(
        "summarization",
        "Extract every date and its event from this text: 'The project "
        "kicked off on 3 March 2025. The first prototype was ready by 14 April, and the beta "
        "launched on 1 June. After feedback, version 1.0 shipped on 20 August 2025.'",
    ),
    hard(
        "summarization",
        "Classify the sentiment of these reviews as positive, negative or "
        "neutral: 1) 'Battery life is amazing.' 2) 'It broke after two days.' 3) 'It does "
        "what it says.'",
    ),
    # --- explanation ---
    hard(
        "explanation",
        "Explain how the transformer attention mechanism works, including "
        "queries, keys and values, with a small numeric example.",
        dups=[
            "explain transformer self attention with queries keys and values and a small "
            "numerical example"
        ],
    ),
    hard(
        "explanation",
        "Explain how HTTPS establishes a secure connection, step by step, from "
        "the TLS handshake to encrypted data.",
    ),
    hard(
        "explanation",
        "Explain how garbage collection works in Python, including reference "
        "counting and the cyclic collector.",
    ),
    hard("explanation", "Explain the CAP theorem with real database examples."),
    hard(
        "explanation",
        "Explain how a B-tree index speeds up database queries and when it does not help.",
    ),
    hard(
        "explanation",
        "Explain the difference between processes, threads and coroutines, "
        "with examples of when to use each.",
        dups=["processes vs threads vs coroutines, when to use each, with examples"],
    ),
    hard(
        "explanation",
        "Explain gradient descent and the role of the learning rate, including "
        "what happens if it is too high or too low.",
    ),
    hard(
        "explanation",
        "Explain how backpropagation computes gradients in a neural network, using the chain rule.",
    ),
    hard("explanation", "Explain how consistent hashing works and why it is useful for caches."),
    hard("explanation", "Explain how JWT authentication works and its main security pitfalls."),
    hard(
        "explanation",
        "Explain the difference between L1 and L2 regularization and when to use each.",
        traps=["Explain the difference between L1 and L2 loss and when to use each."],
    ),
    hard("explanation", "Explain how a vector index like HNSW finds nearest neighbours quickly."),
    hard("explanation", "Explain how Python's GIL affects multithreaded programs."),
    hard("explanation", "Explain BM25 and how it differs from dense vector search."),
    hard(
        "explanation",
        "Explain in detail what happens when you type a URL into a browser and press Enter.",
        dups=["what happens when you type a url in the browser and hit enter, in detail"],
    ),
]


def build() -> list[dict]:
    rows: list[dict] = []

    def add(seed: Seed, prompt: str, dup_of=None, trap_of=None) -> str:
        row_id = f"q{len(rows) + 1:04d}"
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "category": seed.category,
                "difficulty": seed.difficulty,
                "dup_of": dup_of,
                "trap_of": trap_of,
            }
        )
        return row_id

    for seed in SEEDS:
        base_id = add(seed, seed.prompt)
        for dup in seed.dups:
            add(seed, dup, dup_of=base_id)
        for trap in seed.traps:
            add(seed, trap, trap_of=base_id)

    # Shuffle (fixed seed) so a replay doesn't always see the original right before its
    # duplicate, then renumber so ids follow file order.
    random.Random(42).shuffle(rows)
    new_ids = {row["id"]: f"q{i + 1:04d}" for i, row in enumerate(rows)}
    for row in rows:
        row["id"] = new_ids[row["id"]]
        for ref in ("dup_of", "trap_of"):
            if row[ref]:
                row[ref] = new_ids[row[ref]]
    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dups = sum(r["dup_of"] is not None for r in rows)
    traps = sum(r["trap_of"] is not None for r in rows)
    hard_count = sum(r["difficulty"] == "hard" for r in rows)
    print(f"Wrote {len(rows)} prompts to {OUT}")
    print(f"  near-duplicates: {dups} ({dups / len(rows):.0%})")
    print(f"  traps:           {traps}")
    print(f"  hard:            {hard_count} ({hard_count / len(rows):.0%})")


if __name__ == "__main__":
    main()
