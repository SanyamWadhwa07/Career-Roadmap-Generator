import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Read the original dataset
df = pd.read_csv('final_dataset.csv')

# Analyze the existing data structure
domains = df['Domain'].unique().tolist()
levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
all_months = list(range(1, 13))

# Create a more comprehensive structure of domain knowledge
domain_info = {
    "Machine Learning": {
        "Beginner": {
            "skills": [
                "Python Basics", "Data Analysis with Pandas", "NumPy Fundamentals", "Data Visualization", 
                "Statistics Basics", "Probability Theory", "Linear Algebra Fundamentals", "Introduction to ML Algorithms"
            ],
            "resources": [
                ("CS50 Python", "Automate the Boring Stuff"),
                ("Data Science with Python", "Python for Data Analysis"),
                ("NumPy Guide", "Python Data Science Handbook"),
                ("Matplotlib & Seaborn", "Data Visualization with Python"),
                ("Statistics for Data Science", "Think Stats"),
                ("Introduction to Probability", "Probability for Data Science"),
                ("Linear Algebra for ML", "Essence of Linear Algebra"),
                ("Introduction to ML", "ML Crash Course by Google")
            ],
            "projects": [
                "Build a simple calculator", "Analyze COVID-19 data", "Implement array operations from scratch",
                "Create interactive visualizations", "Analyze dataset with statistical methods",
                "Implement probability simulations", "Apply linear algebra in image processing",
                "Build a basic classification model"
            ]
        },
        "Intermediate": {
            "skills": [
                "Supervised Learning Advanced", "Unsupervised Learning", "Feature Engineering", 
                "Model Evaluation", "Decision Trees", "SVM and KNN", "Neural Networks Basics", "Ensemble Methods"
            ],
            "resources": [
                ("StatQuest ML", "Hands-On ML with Scikit-Learn"),
                ("Stanford ML Course", "Hands-On ML with Scikit-Learn"),
                ("Feature Engineering for ML", "Applied ML in Python"),
                ("Model Evaluation Metrics", "Applied Predictive Modeling"),
                ("Decision Trees and Random Forests", "ML in Action"),
                ("SVM and KNN Algorithms", "Pattern Recognition"),
                ("Neural Networks from Scratch", "Deep Learning by Ian Goodfellow"),
                ("Ensemble Methods in ML", "Applied Predictive Modeling")
            ],
            "projects": [
                "Build a classification model for customer churn", "Cluster customer segments", 
                "Optimize model performance through feature selection", "Create a model evaluation pipeline",
                "Predict housing prices with Decision Trees", "Implement image classification with KNN",
                "Build a neural network from scratch", "Ensemble multiple models for improved accuracy"
            ]
        },
        "Advanced": {
            "skills": [
                "Deep Learning Fundamentals", "CNNs and Computer Vision", "RNNs and NLP", 
                "Reinforcement Learning", "GANs", "MLOps & Deployment", "Bayesian ML", "Advanced Optimization"
            ],
            "resources": [
                ("Deep Learning Specialization", "PyTorch Tutorials"),
                ("CS231n Stanford", "Deep Learning for Computer Vision"),
                ("CS224n Stanford", "Natural Language Processing with Deep Learning"),
                ("RL Course by Hugging Face", "Deep RL by OpenAI"),
                ("GAN Specialization", "Generative Deep Learning"),
                ("ML System Design", "TFX Tutorials"),
                ("Bayesian Methods for Hackers", "Probabilistic Programming"),
                ("Advanced Optimization for ML", "Convex Optimization")
            ],
            "projects": [
                "Create an image classifier", "Build an object detection system", 
                "Develop a sentiment analysis model", "Develop a game-playing AI",
                "Generate synthetic images with GANs", "Deploy and monitor a production ML system",
                "Implement Bayesian neural networks", "Optimize complex deep learning architectures"
            ]
        },
        "Expert": {
            "skills": [
                "Transformers and BERT", "Meta-Learning", "Graph Neural Networks", 
                "Distributed ML Systems", "AutoML and Neural Architecture Search", 
                "Quantum Machine Learning", "ML Research", "Multimodal ML"
            ],
            "resources": [
                ("Attention Is All You Need", "Transformer Models"),
                ("Meta-Learning Papers", "Learning to Learn"),
                ("Graph Representation Learning", "Graph Neural Networks"),
                ("Distributed ML", "Large Scale ML"),
                ("AutoML Handbook", "Neural Architecture Search"),
                ("Quantum ML Introduction", "Quantum Algorithms"),
                ("ML Research Papers", "NeurIPS Proceedings"),
                ("Multimodal Deep Learning", "Vision-Language Models")
            ],
            "projects": [
                "Fine-tune transformers for specialized tasks", "Implement few-shot learning systems", 
                "Develop graph-based recommendation systems", "Build distributed training pipelines",
                "Create neural architecture search algorithms", "Explore quantum ML applications",
                "Reproduce and extend research papers", "Build multimodal systems (text-image-audio)"
            ]
        }
    },
    "Web Development": {
        "Beginner": {
            "skills": [
                "HTML & CSS Basics", "CSS Layouts & Responsive Design", "JavaScript Fundamentals", 
                "DOM Manipulation", "Basic Web Design", "Frontend Tooling", "Version Control", "Web Accessibility Basics"
            ],
            "resources": [
                ("CS50 Web", "HTML & CSS by Jon Duckett"),
                ("CSS Grid by Wes Bos", "Responsive Web Design"),
                ("The Odin Project", "Eloquent JavaScript"),
                ("JavaScript.info", "DOM Enlightenment"),
                ("Web Design Principles", "Don't Make Me Think"),
                ("NPM & Webpack", "Modern Frontend Tooling"),
                ("Git and GitHub", "Version Control with Git"),
                ("A11y Project", "Web Accessibility Fundamentals")
            ],
            "projects": [
                "Build a portfolio website", "Create a responsive landing page", 
                "Build an interactive form", "Create a dynamic web page",
                "Design and implement a blog layout", "Set up a modern development environment",
                "Collaborate on a small project using Git", "Make an accessible web component"
            ]
        },
        "Intermediate": {
            "skills": [
                "JavaScript ES6+ Features", "Frontend Frameworks Basics", "State Management", 
                "API Integration", "React.js Advanced", "Backend Basics with Node.js", 
                "Testing Web Applications", "Progressive Web Apps"
            ],
            "resources": [
                ("ES6 for Everyone", "You Don't Know JS"),
                ("React/Vue/Angular Docs", "Frontend Masters"),
                ("Redux/Vuex/NgRx", "State Management Patterns"),
                ("RESTful API Design", "API Design Patterns"),
                ("Fullstack Open", "React Patterns"),
                ("Node.js: The Complete Guide", "Express.js Documentation"),
                ("Testing JavaScript", "Testing Library"),
                ("PWA Workshop", "Progressive Web Apps")
            ],
            "projects": [
                "Build a browser-based game", "Create a single-page application", 
                "Create a social media dashboard", "Build a weather app with external API",
                "Create a complex React application", "Build a REST API",
                "Implement comprehensive tests for web apps", "Convert a website to PWA"
            ]
        },
        "Advanced": {
            "skills": [
                "Authentication & Authorization", "Performance Optimization", "Server-Side Rendering", 
                "GraphQL", "Full-Stack Development", "Deployment & DevOps", "Microservices Architecture", 
                "Web Security"
            ],
            "resources": [
                ("OAuth 2.0 Simplified", "JWT Handbook"),
                ("High Performance Web Sites", "Web Performance in Action"),
                ("Next.js/Nuxt.js Docs", "Server-Side Rendering"),
                ("GraphQL Documentation", "The Road to GraphQL"),
                ("Full Stack Open", "MERN Stack Guide"),
                ("Docker for Developers", "AWS for Frontend Engineers"),
                ("Microservices Architecture", "Building Microservices"),
                ("Web Security Handbook", "OWASP Top 10")
            ],
            "projects": [
                "Implement a secure login system", "Optimize a web application for performance", 
                "Build a SSR application", "Create a GraphQL API",
                "Create an e-commerce platform", "Deploy a MERN app with CI/CD",
                "Design and implement microservices", "Conduct a security audit of web applications"
            ]
        },
        "Expert": {
            "skills": [
                "WebAssembly", "Advanced Browser APIs", "Distributed Web Applications", 
                "Web3 & DApps", "Frontend Architecture", "Internationalization", 
                "Web Components", "Design Systems"
            ],
            "resources": [
                ("WebAssembly in Action", "WebAssembly MDN"),
                ("Advanced Browser APIs", "JavaScript Cookbook"),
                ("Distributed Systems for Web Developers", "Building Scalable Web Apps"),
                ("Full Stack Web3 Development", "Ethereum DApp Development"),
                ("Micro Frontends", "Frontend Architecture for Design Systems"),
                ("i18n & l10n", "Building Multilingual Applications"),
                ("Web Components in Action", "Open Web Components"),
                ("Design Systems Handbook", "Atomic Design")
            ],
            "projects": [
                "Implement performance-critical features in WebAssembly", "Create applications using latest Web APIs", 
                "Build globally distributed web applications", "Create a decentralized application",
                "Implement a micro-frontend architecture", "Create a multilingual global website",
                "Build a library of reusable web components", "Design and implement a comprehensive design system"
            ]
        }
    },
    "Software Development": {
        "Beginner": {
            "skills": [
                "Programming Fundamentals", "Data Structures", "Algorithms Basics", 
                "Command Line", "Version Control", "Software Development Lifecycle", 
                "Basic Project Management", "Programming Language Basics"
            ],
            "resources": [
                ("CS50 Introduction to CS", "Programming from the Ground Up"),
                ("Data Structures & Algorithms in Python", "Grokking Algorithms"),
                ("Intro to Algorithms", "AlgoExpert"),
                ("The Linux Command Line", "Command Line Crash Course"),
                ("Pro Git", "Git Immersion"),
                ("Software Engineering", "Clean Code"),
                ("Agile Fundamentals", "Scrum Guide"),
                ("Python/Java/C++ Tutorials", "Language Documentation")
            ],
            "projects": [
                "Solve algorithm challenges", "Implement common data structures", 
                "Implement sorting algorithms", "Create command line applications",
                "Set up Git workflows", "Follow SDLC for a small project",
                "Manage a project using agile methodology", "Build small programs in chosen language"
            ]
        },
        "Intermediate": {
            "skills": [
                "Testing Fundamentals", "Object-Oriented Programming", "Design Patterns", 
                "Debugging Techniques", "Database Integration", "Build Tools & Package Management", 
                "Continuous Integration", "Code Quality"
            ],
            "resources": [
                ("Test-Driven Development", "The Art of Unit Testing"),
                ("Java Masterclass", "Head First Java"),
                ("Design Patterns by GoF", "Refactoring Guru"),
                ("Debugging with GDB", "Chrome DevTools"),
                ("SQL & NoSQL", "Database Systems"),
                ("Maven/Gradle/NPM", "Build Tool Documentation"),
                ("Jenkins/Github Actions", "CI/CD Pipeline"),
                ("Clean Code", "Code Complete")
            ],
            "projects": [
                "Write tests for a library", "Build a Library Management System", 
                "Implement common design patterns", "Debug and fix complex issues",
                "Create applications with database backends", "Set up build automation",
                "Implement CI pipeline", "Refactor code for quality"
            ]
        },
        "Advanced": {
            "skills": [
                "Clean Code & Refactoring", "Advanced Design Patterns", "Concurrency & Parallelism", 
                "Microservices", "Distributed Systems", "System Design & Architecture", 
                "Performance Optimization", "Security Best Practices"
            ],
            "resources": [
                ("Clean Code by Robert Martin", "Refactoring by Martin Fowler"),
                ("Patterns of Enterprise Application Architecture", "Domain-Driven Design"),
                ("Java Concurrency in Practice", "Python Concurrency"),
                ("Building Microservices", "Microservices Patterns"),
                ("Designing Data-Intensive Applications", "Distributed Systems for Practitioners"),
                ("System Design Interview", "Scalability for Dummies"),
                ("High Performance Code", "Writing Efficient Programs"),
                ("Secure Coding", "OWASP Guide")
            ],
            "projects": [
                "Refactor a legacy codebase", "Apply advanced patterns in large systems", 
                "Build a multithreaded application", "Design and implement microservices",
                "Design a chat application", "Design a Scalable E-Commerce System",
                "Optimize application performance", "Implement security best practices"
            ]
        },
        "Expert": {
            "skills": [
                "Software Architecture", "Technical Leadership", "Advanced System Design", 
                "High-Scale Systems", "Custom Languages & Compilers", "Low-Level Programming", 
                "AI-Assisted Development", "Research & Innovation"
            ],
            "resources": [
                ("Software Architecture in Practice", "Clean Architecture"),
                ("The Staff Engineer's Path", "Technical Leadership"),
                ("System Design Interview: Volume 2", "Google's System Design"),
                ("Designing Data-Intensive Applications", "Web Scalability"),
                ("Programming Language Implementation", "Compilers: Principles & Practice"),
                ("C/Assembly Programming", "Computer Systems: A Programmer's Perspective"),
                ("AI-Assisted Coding", "GitHub Copilot Guide"),
                ("Research Papers", "Innovation in Software Engineering")
            ],
            "projects": [
                "Design enterprise architecture", "Lead technical initiatives", 
                "Design systems handling millions of users", "Implement globally distributed systems",
                "Create a domain-specific language", "Optimize at hardware level",
                "Build AI-powered development tools", "Contribute to open source or research"
            ]
        }
    },
    "DevOps": {
        "Beginner": {
            "skills": [
                "Linux Basics", "Linux & Command Line", "Version Control with Git", 
                "CI Fundamentals", "Containerization Basics", "Basic Cloud Concepts", 
                "Monitoring Fundamentals", "Networking Basics"
            ],
            "resources": [
                ("Linux Journey", "The Linux Command Line by William Shotts"),
                ("Linux Command Line Basics", "Linux Administration Handbook"),
                ("Git Pro", "GitHub Training"),
                ("GitHub Actions/Jenkins/GitLab CI", "CI/CD for DevOps"),
                ("Docker Fundamentals", "Docker Deep Dive"),
                ("Cloud Computing Fundamentals", "AWS/Azure/GCP Basics"),
                ("Monitoring Tools Overview", "Prometheus Getting Started"),
                ("Computer Networking", "Networking for DevOps")
            ],
            "projects": [
                "Set up a Linux development environment", "Set up and manage a Linux server", 
                "Set up Git workflows and hooks", "Create a basic CI pipeline",
                "Containerize a web application", "Deploy a simple app to the cloud",
                "Set up basic monitoring", "Configure network settings for applications"
            ]
        },
        "Intermediate": {
            "skills": [
                "Containerization Advanced", "Kubernetes Basics", "Infrastructure as Code", 
                "Configuration Management", "Cloud Services", "CI/CD Pipelines", 
                "Monitoring & Logging", "Security in DevOps"
            ],
            "resources": [
                ("Mastering Docker", "Docker in Practice"),
                ("Kubernetes Up & Running", "Kubernetes in Action"),
                ("Terraform Up & Running", "Infrastructure as Code by Kief Morris"),
                ("Ansible for DevOps", "Puppet Beginner's Guide"),
                ("AWS/Azure/GCP Services", "Cloud Architecture Patterns"),
                ("Continuous Delivery", "Jenkins: The Definitive Guide"),
                ("The Art of Monitoring", "Elasticsearch, Logstash, and Kibana"),
                ("DevSecOps", "Security in DevOps Pipelines")
            ],
            "projects": [
                "Build multi-container applications", "Deploy an app to Kubernetes", 
                "Provision cloud infrastructure with Terraform", "Automate server configuration",
                "Set up cloud-native services", "Create advanced CI/CD pipelines",
                "Implement logging and monitoring solutions", "Implement security in CI/CD"
            ]
        },
        "Advanced": {
            "skills": [
                "Advanced Kubernetes", "Advanced Infrastructure as Code", "Multi-Cloud Strategies", 
                "Site Reliability Engineering", "CI/CD & Infrastructure as Code", 
                "Service Mesh", "Cloud Native Architecture", "DevOps Culture & Practices"
            ],
            "resources": [
                ("Kubernetes Patterns", "Production Kubernetes"),
                ("Terraform: Up & Running 2nd Edition", "Infrastructure as Code Cookbook"),
                ("Multi-Cloud Architecture", "Cloud Native Patterns"),
                ("Site Reliability Engineering", "The SRE Workbook"),
                ("GitLab CI/CD Course", "Continuous Delivery"),
                ("Istio in Action", "Service Mesh Patterns"),
                ("Cloud Native Architecture", "Kubernetes Patterns"),
                ("The DevOps Handbook", "Accelerate")
            ],
            "projects": [
                "Set up advanced Kubernetes features", "Implement complex IaC workflows", 
                "Create multi-cloud deployment strategy", "Implement SRE practices",
                "Set up an end-to-end CI/CD pipeline", "Implement service mesh for microservices",
                "Design cloud-native applications", "Transform organization with DevOps"
            ]
        },
        "Expert": {
            "skills": [
                "GitOps", "Chaos Engineering", "Advanced Observability", 
                "Platform Engineering", "Cloud Cost Optimization", "Cloud Security", 
                "FinOps", "Advanced Automation"
            ],
            "resources": [
                ("GitOps and Kubernetes", "Flux & ArgoCD"),
                ("Chaos Engineering", "Chaos Monkey Guide"),
                ("Observability Engineering", "Distributed Systems Observability"),
                ("Platform Engineering", "Internal Developer Platforms"),
                ("Cloud Cost Optimization", "FinOps Guide"),
                ("Cloud Security Patterns", "Security in the Cloud"),
                ("FinOps - Cloud Financial Management", "Cloud FinOps"),
                ("Advanced Automation Techniques", "Infrastructure Automation at Scale")
            ],
            "projects": [
                "Implement GitOps workflows", "Create chaos engineering tests", 
                "Build advanced observability platform", "Create an internal developer platform",
                "Optimize cloud costs at scale", "Implement advanced cloud security",
                "Establish FinOps practices", "Create advanced automation frameworks"
            ]
        }
    },
    "Data Analyst": {
        "Beginner": {
            "skills": [
                "Data Analysis Basics", "SQL & Data Querying", "Data Cleaning", 
                "Excel for Data Analysis", "Basic Statistics", "Data Collection", 
                "Data Types & Formats", "Exploratory Data Analysis"
            ],
            "resources": [
                ("Data Analysis with Python", "Analyzing Data with Excel"),
                ("SQL for Data Analysis", "Practical SQL"),
                ("Data Cleaning Handbook", "Data Preparation for Analytics"),
                ("Microsoft Excel for Data Analysis", "Excel Data Analysis by Example"),
                ("Statistics for Data Science", "Think Stats"),
                ("Data Collection Methods", "Web Scraping with Python"),
                ("Data Formats & ETL", "Learning Data Formats"),
                ("Exploratory Data Analysis", "Practical Statistics for Data Scientists")
            ],
            "projects": [
                "Clean and analyze a dataset", "Analyze business data with SQL queries", 
                "Prepare messy data for analysis", "Create Excel dashboards",
                "Apply statistical methods to data", "Collect and structure data",
                "Work with various data formats", "Perform EDA on a complex dataset"
            ]
        },
        "Intermediate": {
            "skills": [
                "Data Visualization Basics", "Data Visualization & Dashboard Creation", 
                "Intermediate SQL", "Python for Data Analysis", "Business Intelligence Tools", 
                "Statistical Analysis", "Data Storytelling", "ETL Processes"
            ],
            "resources": [
                ("Data Visualization with Python", "Tableau Fundamentals"),
                ("Tableau Fundamentals", "Storytelling with Data"),
                ("Advanced SQL for Data Analysis", "SQL Cookbook"),
                ("Python for Data Analysis", "Pandas for Everyone"),
                ("Power BI/Tableau/Looker", "Data Visualization Tools"),
                ("Statistical Inference", "Practical Statistics for Data Scientists"),
                ("Storytelling with Data", "Data Points"),
                ("ETL with Python", "Data Pipelines")
            ],
            "projects": [
                "Create basic data visualizations", "Create an interactive business dashboard", 
                "Write complex SQL queries", "Analyze data with Python",
                "Build BI reports and dashboards", "Apply statistical tests to data",
                "Create data stories for stakeholders", "Build ETL pipelines"
            ]
        },
        "Advanced": {
            "skills": [
                "Advanced SQL & Data Warehousing", "Advanced Python for Data Analysis", 
                "A/B Testing", "Statistical Modeling", "Statistical Analysis & Hypothesis Testing", 
                "Big Data Analysis", "Machine Learning for Analysts", "BI Architecture"
            ],
            "resources": [
                ("SQL for Data Warehousing", "Kimball's Data Warehouse Toolkit"),
                ("Python Data Science Handbook", "Efficient Python for Data Analysis"),
                ("Trustworthy Online Controlled Experiments", "Experimentation Works"),
                ("Statistical Learning", "Data Analysis with R"),
                ("Business Statistics Course", "Naked Statistics by Charles Wheelan"),
                ("Big Data Analysis", "Spark for Data Analysis"),
                ("Hands-On ML for Data Analysts", "Machine Learning for Business"),
                ("BI Architecture", "Data Warehouse Architecture")
            ],
            "projects": [
                "Design a data warehouse schema", "Create advanced data analysis programs", 
                "Design and analyze A/B tests", "Build predictive models for business",
                "Conduct A/B tests for business optimization", "Analyze big data sets",
                "Apply ML to business problems", "Design BI architecture"
            ]
        },
        "Expert": {
            "skills": [
                "Data Strategy", "Advanced Data Architecture", "Experimental Design", 
                "Causal Inference", "Advanced Business Analytics", "AI for Analytics", 
                "Analytics Leadership", "Data Ethics & Governance"
            ],
            "resources": [
                ("Data Strategy", "Creating a Data-Driven Organization"),
                ("Modern Data Architecture", "Data Mesh"),
                ("Design of Experiments", "Causal Inference"),
                ("Causal Inference in Python", "The Book of Why"),
                ("Business Analytics", "Data Science for Business"),
                ("AI for Analytics", "Automated Machine Learning"),
                ("Leading Data Teams", "Analytics Leadership"),
                ("Data Ethics", "Data Governance Framework")
            ],
            "projects": [
                "Develop organizational data strategy", "Design modern data architecture", 
                "Design complex experiments", "Implement causal inference models",
                "Create advanced business analytics", "Implement AI in analytics workflows",
                "Lead data analytics initiatives", "Establish data governance framework"
            ]
        }
    },
    "Cybersecurity": {
        "Beginner": {
            "skills": [
                "Networking Basics", "Linux for Security", "Cybersecurity Fundamentals", 
                "Security Concepts", "Basic Cryptography", "Security Tools", 
                "Security Policies", "Information Security Basics"
            ],
            "resources": [
                ("Computer Networking by Stanford", "Computer Networking: Kurose & Ross"),
                ("Linux Basics for Hackers", "The Linux Command Line"),
                ("CompTIA Security+", "Cybersecurity Fundamentals"),
                ("Introduction to Cybersecurity", "Cybersecurity Basics"),
                ("Applied Cryptography", "Cryptography 101"),
                ("Kali Linux Tools", "Security Tool Basics"),
                ("Security Policy Development", "NIST Cybersecurity Framework"),
                ("Information Security Handbook", "Principles of Information Security")
            ],
            "projects": [
                "Set up a Virtual Private Network (VPN)", "Harden a Linux server", 
                "Perform basic security assessment", "Implement security controls",
                "Implement basic encryption", "Use security tools for assessment",
                "Create a security policy", "Conduct a risk assessment"
            ]
        },
        "Intermediate": {
            "skills": [
                "Web Security", "Network Security", "Ethical Hacking", 
                "Digital Forensics", "Security Monitoring", "Vulnerability Assessment", 
                "Security Compliance", "Penetration Testing Basics"
            ],
            "resources": [
                ("Web Security Academy", "OWASP Top 10"),
                ("Network Security Essentials", "Practical Packet Analysis"),
                ("Certified Ethical Hacker (CEH)", "The Web Application Hacker's Handbook"),
                ("Digital Forensics Basics", "Computer Forensics"),
                ("Security Information and Event Management", "Security Monitoring"),
                ("Vulnerability Assessment", "Web Application Security"),
                ("Security Compliance Standards", "GDPR & HIPAA"),
                ("Penetration Testing", "Metasploit Unleashed")
            ],
            "projects": [
                "Find vulnerabilities in a web app", "Configure a firewall and IDS", 
                "Perform penetration testing on a website", "Analyze digital evidence",
                "Set up security monitoring", "Conduct vulnerability assessments",
                "Implement compliance controls", "Perform basic penetration tests"
            ]
        },
        "Advanced": {
            "skills": [
                "Malware Analysis", "Threat Hunting", "Red Team Operations", 
                "Security Architecture", "Security Operations & Incident Response", 
                "Advanced Penetration Testing", "Cloud Security", "Secure Development"
            ],
            "resources": [
                ("Practical Malware Analysis", "Learning Malware Analysis"),
                ("Threat Hunting with Elastic Stack", "Applied Network Security Monitoring"),
                ("Red Team Field Manual", "Advanced Penetration Testing"),
                ("CISSP Study Guide", "Zero Trust Networks"),
                ("CompTIA Security+", "Incident Response & Computer Forensics"),
                ("Advanced Penetration Testing", "The Hacker Playbook"),
                ("AWS/Azure/GCP Security", "Cloud Security Alliance"),
                ("Secure Coding", "OWASP SAMM")
            ],
            "projects": [
                "Analyze a malware sample in a sandbox environment", "Create threat detection rules", 
                "Conduct a red team exercise", "Design a secure enterprise architecture",
                "Create an incident response playbook", "Perform advanced penetration testing",
                "Implement cloud security controls", "Create secure development guidelines"
            ]
        },
        "Expert": {
            "skills": [
                "Advanced Threat Intelligence", "Security Research", "Exploit Development", 
                "Security Program Management", "Advanced Cryptography", "Security Engineering", 
                "Critical Infrastructure Security", "Advanced Defensive Security"
            ],
            "resources": [
                ("Intelligence-Driven Incident Response", "Threat Intelligence"),
                ("Security Research Methodologies", "Security Research Papers"),
                ("The Shellcoder's Handbook", "Hacking: The Art of Exploitation"),
                ("CISO Desk Reference", "Security Program Management"),
                ("Cryptography Engineering", "Modern Cryptography"),
                ("Security Engineering", "Building Secure Systems"),
                ("Industrial Control Systems Security", "Critical Infrastructure Protection"),
                ("Blue Team Handbook", "Defensive Security Architecture")
            ],
            "projects": [
                "Develop threat intelligence program", "Conduct security research", 
                "Develop secure exploits ethically", "Design enterprise security program",
                "Implement advanced cryptographic systems", "Design secure systems",
                "Secure industrial control systems", "Implement advanced defensive measures"
            ]
        }
    }
}

# Define typical month ranges for different levels
level_month_ranges = {
    "Beginner": list(range(1, 5)),      # Months 1-4
    "Intermediate": list(range(4, 9)),  # Months 4-8
    "Advanced": list(range(7, 13)),     # Months 7-12
    "Expert": list(range(9, 13)) + [1, 2]  # Months 9-12 and 1-2 (wrapping around)
}

# Create new rows
new_rows = []

# For each domain, create a comprehensive learning path
for domain in domain_info.keys():
    for level in domain_info[domain].keys():
        # Select appropriate months for this level
        valid_months = level_month_ranges[level]
        
        # Get the skills, resources, and projects for this domain and level
        skills = domain_info[domain][level]["skills"]
        resources = domain_info[domain][level]["resources"]
        projects = domain_info[domain][level]["projects"]
        
        # Create entries for each skill in this domain and level
        for i, skill in enumerate(skills):
            # Choose an appropriate month
            month = valid_months[i % len(valid_months)]
            
            # Get corresponding resources and project
            resource_pair = resources[i % len(resources)]
            project = projects[i % len(projects)]
            
            # Add the row
            new_rows.append({
                "Domain": domain,
                "Month": month,
                "Level": level,
                "Skill": skill,
                "Recommended Resource 1": resource_pair[0],
                "Recommended Resource 2": resource_pair[1],
                "Project": project
            })

# Additional domains with fewer entries to add variety
additional_domains = {
    "AI Engineer": {
        "Beginner": {
            "skills": ["Python for AI", "Python & Deep Learning Basics", "Data Structures for AI", "Math for AI"],
            "resources": [
                ("Python for Data Science", "Think Python"),
                ("PyTorch Fundamentals", "Hands-On Machine Learning"),
                ("Algorithms for AI", "Data Structures in Python"),
                ("Mathematics for ML", "Linear Algebra for AI")
            ],
            "projects": [
                "Build data processing scripts", 
                "Build a neural network from scratch",
                "Implement efficient data structures for ML",
                "Apply mathematical concepts in AI applications"
            ]
        },
        "Intermediate": {
            "skills": ["Deep Learning Frameworks", "Computer Vision Basics", "NLP Fundamentals", "Model Deployment & MLOps"],
            "resources": [
                ("TensorFlow in Practice", "Deep Learning with PyTorch"),
                ("Computer Vision Basics", "OpenCV with Python"),
                ("NLP with Python", "Natural Language Processing"),
                ("MLOps Specialization", "Building Machine Learning Powered Applications")
            ],
            "projects": [
                "Create custom neural network architectures",
                "Build an image recognition system",
                "Create a text classification system",
                "Deploy an ML model as a REST API"
            ]
        },
        "Advanced": {
            "skills": ["ML System Design", "Model Optimization", "Edge AI Deployment", "AI Infrastructure"],
            "resources": [
                ("Machine Learning System Design", "ML Engineering for Production"),
                ("High Performance ML", "TensorRT Programming Guide"),
                ("TinyML", "AI at the Edge"),
                ("ML at Scale", "Design Patterns for High-Available Systems")
            ],
            "projects": [
                "Design scalable ML systems",
                "Optimize models for inference",
                "Deploy models to edge devices",
                "Build AI infrastructure for large models"
            ]
        }
    },
    "Blockchain": {
        "Beginner": {
            "skills": ["Blockchain Theory", "Blockchain Fundamentals", "Cryptocurrency Basics", "Basic Smart Contracts"],
            "resources": [
                ("Blockchain Basics", "Mastering Bitcoin"),
                ("Blockchain Technology", "Blockchain Revolution"),
                ("Introduction to Cryptocurrencies", "Digital Gold"),
                ("Smart Contracts 101", "Introduction to Ethereum")
            ],
            "projects": [
                "Create a simple blockchain in Python",
                "Create a cryptocurrency wallet interface",
                "Analyze crypto market data",
                "Deploy a basic smart contract"
            ]
        },
        "Intermediate": {
            "skills": ["Smart Contract Development", "DeFi Fundamentals", "Web3 Development", "Blockchain Security"],
            "resources": [
                ("Solidity Programming", "Ethereum Smart Contract Development"),
                ("DeFi Handbook", "How to DeFi"),
                ("Web3.js Guide", "Full Stack Ethereum"),
                ("Secure Smart Contracts", "Blockchain Security")
            ],
            "projects": [
                "Build a decentralized application",
                "Create a DeFi protocol interface",
                "Develop a Web3 frontend",
                "Audit smart contracts for vulnerabilities"
            ]
        },
        "Advanced": {
            "skills": ["Consensus Mechanisms", "Layer 2 Solutions", "Cross-Chain Development", "Tokenomics Design"],
            "resources": [
                ("Advanced Consensus", "Blockchain Consensus Algorithms"),
                ("Layer 2 Scaling", "Optimistic Rollups & ZK Proofs"),
                ("Cross-Chain Protocols", "Interoperability in Blockchain"),
                ("Token Economics", "Designing Token Economies")
            ],
            "projects": [
                "Implement a consensus algorithm",
                "Build a Layer 2 solution",
                "Create a cross-chain bridge",
                "Design a token economic model"
            ]
        }
    },
    "Game Development": {
        "Beginner": {
            "skills": ["Game Engines Basics", "2D Game Development", "Game Design Fundamentals", "Game Mathematics"],
            "resources": [
                ("Unity Getting Started", "Godot Engine Basics"),
                ("2D Game Development", "Pixel Art for Games"),
                ("Game Design", "The Art of Game Design"),
                ("Game Mathematics", "Math for Game Developers")
            ],
            "projects": [
                "Create a simple game in Unity/Godot",
                "Build a 2D platformer",
                "Design a game concept document",
                "Implement basic game physics"
            ]
        },
        "Intermediate": {
            "skills": ["3D Game Development", "Game Physics", "Game AI", "Game Networking"],
            "resources": [
                ("3D Game Programming", "3D Math for Game Programming"),
                ("Game Physics Engine Development", "Physics for Game Developers"),
                ("Game AI Pro", "Artificial Intelligence for Games"),
                ("Multiplayer Game Programming", "Game Networking")
            ],
            "projects": [
                "Develop a 3D game environment",
                "Create custom physics for a game",
                "Implement AI for NPCs",
                "Build a multiplayer game prototype"
            ]
        },
        "Advanced": {
            "skills": ["Game Engine Architecture", "Advanced Graphics Programming", "Game Optimization", "Procedural Generation"],
            "resources": [
                ("Game Engine Architecture", "Game Programming Patterns"),
                ("Graphics Programming", "Real-Time Rendering"),
                ("High Performance Games", "Optimizing Games"),
                ("Procedural Content Generation", "Procedural Generation in Games")
            ],
            "projects": [
                "Create a custom game engine component",
                "Implement advanced rendering techniques",
                "Optimize a game for performance",
                "Create procedural level generation"
            ]
        }
    }
}

# For each additional domain, add entries to the dataset
for domain in additional_domains.keys():
    for level in additional_domains[domain].keys():
        # Select appropriate months for this level
        valid_months = level_month_ranges[level]
        
        # Get the skills, resources, and projects for this domain and level
        skills = additional_domains[domain][level]["skills"]
        resources = additional_domains[domain][level]["resources"]
        projects = additional_domains[domain][level]["projects"]
        
        # Create entries for each skill in this domain and level
        for i, skill in enumerate(skills):
            # Choose an appropriate month
            month = valid_months[i % len(valid_months)]
            
            # Get corresponding resources and project
            resource_pair = resources[i % len(resources)]
            project = projects[i % len(projects)]
            
            # Add the row
            new_rows.append({
                "Domain": domain,
                "Month": month,
                "Level": level,
                "Skill": skill,
                "Recommended Resource 1": resource_pair[0],
                "Recommended Resource 2": resource_pair[1],
                "Project": project
            })

# Convert rows to DataFrame
enhanced_df = pd.DataFrame(new_rows)

# Add some realistic variations for better dataset quality
# Add specific dates
enhanced_df['Date'] = enhanced_df.apply(lambda x: f"2023-{x['Month']:02d}-{random.randint(1, 28):02d}", axis=1)

# Add estimated study hours
enhanced_df['Estimated Hours'] = enhanced_df.apply(
    lambda x: random.randint(10, 20) if x['Level'] == 'Beginner' 
              else random.randint(20, 40) if x['Level'] == 'Intermediate'
              else random.randint(40, 80) if x['Level'] == 'Advanced'
              else random.randint(80, 120),
    axis=1
)

# Add prerequisite skills
prerequisite_skills = {
    "Beginner": ["None", "Basic computer literacy", "Problem-solving"],
    "Intermediate": ["Fundamentals in the domain", "Basic programming", "Previous level skills"],
    "Advanced": ["Strong foundation in the domain", "Practical experience", "Technical background"],
    "Expert": ["Years of experience", "Industry knowledge", "Domain mastery"]
}

enhanced_df['Prerequisites'] = enhanced_df.apply(
    lambda x: random.choice(prerequisite_skills[x['Level']]), 
    axis=1
)

# Add learning outcomes
outcome_templates = {
    "Beginner": [
        "Understand the basics of {skill}",
        "Build foundation in {skill}",
        "Create simple projects using {skill}",
        "Recognize key concepts in {skill}"
    ],
    "Intermediate": [
        "Apply {skill} to solve real problems",
        "Develop intermediate-level applications with {skill}",
        "Analyze and improve {skill} implementations",
        "Create efficient solutions using {skill}"
    ],
    "Advanced": [
        "Master complex aspects of {skill}",
        "Design professional-level systems with {skill}",
        "Optimize and scale solutions using {skill}",
        "Lead projects involving {skill}"
    ],
    "Expert": [
        "Innovate in the field of {skill}",
        "Architect enterprise solutions using {skill}",
        "Research and develop new approaches to {skill}",
        "Teach and mentor others in {skill}"
    ]
}

enhanced_df['Learning Outcome'] = enhanced_df.apply(
    lambda x: random.choice(outcome_templates[x['Level']]).format(skill=x['Skill']), 
    axis=1
)

# Add difficulty rating
enhanced_df['Difficulty Rating'] = enhanced_df.apply(
    lambda x: random.randint(1, 3) if x['Level'] == 'Beginner' 
              else random.randint(4, 6) if x['Level'] == 'Intermediate'
              else random.randint(7, 8) if x['Level'] == 'Advanced'
              else random.randint(9, 10),
    axis=1
)

# Add community rating
enhanced_df['Community Rating'] = enhanced_df.apply(
    lambda x: round(random.uniform(3.0, 5.0), 1), 
    axis=1
)

# Add success metrics
success_metrics = {
    "Beginner": [
        "Complete project successfully", 
        "Pass basic assessment", 
        "Create a working demo",
        "Understand core concepts"
    ],
    "Intermediate": [
        "Deploy a functional application", 
        "Complete a complex project", 
        "Solve real-world problems",
        "Implement best practices"
    ],
    "Advanced": [
        "Create advanced systems", 
        "Optimize for performance", 
        "Lead project implementation",
        "Publish case study"
    ],
    "Expert": [
        "Innovate in the field", 
        "Create industry standard solution", 
        "Contribute to community knowledge",
        "Develop novel approach"
    ]
}

enhanced_df['Success Metric'] = enhanced_df.apply(
    lambda x: random.choice(success_metrics[x['Level']]), 
    axis=1
)

# Add tags for searchability
def generate_tags(row):
    domain_tag = row['Domain'].lower().replace(' ', '-')
    level_tag = row['Level'].lower()
    skill_tags = row['Skill'].lower().replace(' ', '-').replace('&', 'and')
    return f"{domain_tag},{level_tag},{skill_tags}"

enhanced_df['Tags'] = enhanced_df.apply(generate_tags, axis=1)

# Add a unique ID for each entry
enhanced_df['ID'] = [f"LS{i:04d}" for i in range(1, len(enhanced_df) + 1)]

# Reorder columns
final_columns = [
    'ID', 'Domain', 'Level', 'Skill', 'Month', 'Date', 
    'Estimated Hours', 'Difficulty Rating', 'Community Rating',
    'Prerequisites', 'Recommended Resource 1', 'Recommended Resource 2', 
    'Project', 'Learning Outcome', 'Success Metric', 'Tags'
]

final_df = enhanced_df[final_columns]

# Save to CSV
final_df.to_csv('enhanced_learning_paths.csv', index=False)

# Print sample of the dataset
print(f"Generated {len(final_df)} learning path entries")
print("\nSample of the dataset:")
print(final_df.head())

# Basic statistics
print("\nBasic statistics:")
print(f"Number of domains: {final_df['Domain'].nunique()}")
print(f"Number of skills: {final_df['Skill'].nunique()}")
print(f"Entries by level:")
print(final_df['Level'].value_counts())
print("\nEntries by domain:")
print(final_df['Domain'].value_counts())

# Load the dataset
dataset_path = "./final_dataset.csv"

df = pd.read_csv(dataset_path)

# Function to generate similar skills
def generate_similar_skills(skill):
    variations = [
        f"Advanced {skill}",
        f"{skill} Basics",
        f"{skill} Fundamentals",
        f"{skill} Techniques",
        f"{skill} Applications",
        f"{skill} for Beginners",
        f"{skill} Mastery"
    ]
    return random.choice(variations)

# Generate new rows with similar skills
enriched_rows = []
for _, row in df.iterrows():
    for _ in range(2):  # Generate 2 similar skills per original skill
        new_row = row.copy()
        new_row["Skill"] = generate_similar_skills(row["Skill"])
        enriched_rows.append(new_row)

# Append the new rows to the original dataset
enriched_df = pd.concat([df, pd.DataFrame(enriched_rows)], ignore_index=True)

# Replace 'Other' in the Skill column with variations based on the original_skill column
def replace_other_with_variations(row):
    if row["Skill"] == "Other":
        return generate_similar_skills(row["original_skill"])
    return row["Skill"]

enriched_df["Skill"] = enriched_df.apply(replace_other_with_variations, axis=1)

# Save the updated enriched dataset
enriched_dataset_path = "enriched_dataset_updated.csv"
enriched_df.to_csv(enriched_dataset_path, index=False)
print(f"✓ Updated enriched dataset saved to {enriched_dataset_path}")

# Generate augmented data
def generate_augmented_data(df, num_samples=100):
    augmented_data = []

    for _ in range(num_samples):
        # Randomly select a row from the dataset
        row = df.sample(n=1).iloc[0]

        # Modify the values slightly to create new data
        domain = row['Domain']
        level = random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert'])
        month = max(1, min(12, row['Month'] + random.randint(-2, 2)))  # Adjust month within range
        skill = row['Skill']
        resource1 = row['Recommended Resource 1']
        resource2 = row['Recommended Resource 2']
        project = row['Project']

        # Add the new row to the augmented data
        augmented_data.append({
            'Domain': domain,
            'Month': month,
            'Level': level,
            'Skill': skill,
            'Recommended Resource 1': resource1,
            'Recommended Resource 2': resource2,
            'Project': project
        })

    return pd.DataFrame(augmented_data)

# Generate augmented data
augmented_df = generate_augmented_data(df, num_samples=200)

# Combine with the original dataset
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Save the new dataset
combined_df.to_csv("./dataset_iteration/augmented_dataset.csv", index=False)
print("Augmented dataset saved to dataset_iteration/augmented_dataset.csv")