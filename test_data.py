"""Generate diverse query-passage pairs for reranker evaluation."""


def generate_test_pairs() -> list[dict]:
    """Return ~30 diverse query-passage pairs across relevance categories.

    Each pair has: query, passage, label (high/medium/low/none), category.
    """
    pairs = [
        # === HIGH RELEVANCE: passage directly answers the query ===
        {
            "query": "What is the capital of France?",
            "passage": "Paris is the capital and most populous city of France. It has been the nation's capital since the late 10th century.",
            "label": "high",
            "category": "factual",
        },
        {
            "query": "How does photosynthesis work?",
            "passage": "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs primarily in the chloroplasts of leaf cells using chlorophyll pigments.",
            "label": "high",
            "category": "factual",
        },
        {
            "query": "When was the first moon landing?",
            "passage": "On July 20, 1969, NASA's Apollo 11 mission successfully landed the first humans on the Moon. Astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.",
            "label": "high",
            "category": "factual",
        },
        {
            "query": "What causes earthquakes?",
            "passage": "Earthquakes are caused by the sudden release of energy in the Earth's crust that creates seismic waves. This typically occurs when tectonic plates along fault lines slip past one another or collide.",
            "label": "high",
            "category": "factual",
        },
        {
            "query": "How to sort a list in Python?",
            "passage": "In Python, you can sort a list using the built-in sort() method which sorts in-place, or the sorted() function which returns a new sorted list. Both accept a key parameter for custom sorting and a reverse parameter for descending order.",
            "label": "high",
            "category": "technical",
        },
        {
            "query": "What are the symptoms of diabetes?",
            "passage": "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, slow-healing cuts and wounds, and unexplained weight loss. Type 2 diabetes symptoms often develop gradually over several years.",
            "label": "high",
            "category": "medical",
        },
        {
            "query": "Explain the theory of relativity",
            "passage": "Einstein's theory of relativity consists of two interrelated physics theories: special relativity and general relativity. Special relativity shows that the laws of physics are the same for all non-accelerating observers and that the speed of light is constant. General relativity describes gravity as a curvature of spacetime caused by mass and energy.",
            "label": "high",
            "category": "science",
        },
        {
            "query": "What is machine learning?",
            "passage": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions.",
            "label": "high",
            "category": "technical",
        },

        # === MEDIUM RELEVANCE: topically related but not directly answering ===
        {
            "query": "What is the capital of France?",
            "passage": "France is a country in Western Europe known for its medieval cities, alpine villages and Mediterranean beaches. Its capital, one of the world's major centers of art, fashion, and culture, is famous for the Eiffel Tower and the Louvre Museum.",
            "label": "medium",
            "category": "partial_answer",
        },
        {
            "query": "How does photosynthesis work?",
            "passage": "Plants require several essential nutrients to grow, including nitrogen, phosphorus, and potassium. They absorb these nutrients through their root systems and use sunlight as an energy source for various metabolic processes.",
            "label": "medium",
            "category": "related_topic",
        },
        {
            "query": "What causes earthquakes?",
            "passage": "The San Andreas Fault in California is one of the most studied fault lines in the world. Major earthquakes along this fault have caused significant damage to cities such as San Francisco and Los Angeles throughout history.",
            "label": "medium",
            "category": "related_topic",
        },
        {
            "query": "How to sort a list in Python?",
            "passage": "Python provides many built-in data structures including lists, tuples, sets, and dictionaries. Lists are mutable ordered sequences that support various operations like appending, inserting, and removing elements.",
            "label": "medium",
            "category": "related_topic",
        },
        {
            "query": "What is machine learning?",
            "passage": "The technology industry has seen rapid growth over the past decade, with companies investing heavily in data centers and cloud computing infrastructure. Artificial intelligence research has attracted billions of dollars in funding from major corporations.",
            "label": "medium",
            "category": "related_topic",
        },
        {
            "query": "Explain the theory of relativity",
            "passage": "Albert Einstein was born in Ulm, Germany in 1879. He received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect, and he is widely regarded as one of the most influential physicists of all time.",
            "label": "medium",
            "category": "related_entity",
        },

        # === LOW RELEVANCE: minimal topical overlap ===
        {
            "query": "What is the capital of France?",
            "passage": "The European Union is a political and economic union of 27 member states. It has developed an internal single market through a standardised system of laws that apply in all member states.",
            "label": "low",
            "category": "distant_topic",
        },
        {
            "query": "How does photosynthesis work?",
            "passage": "Solar panels convert sunlight into electrical energy using photovoltaic cells made of semiconductor materials. The efficiency of modern solar panels ranges from 15% to over 22%.",
            "label": "low",
            "category": "distant_topic",
        },
        {
            "query": "What are the symptoms of diabetes?",
            "passage": "Regular exercise has numerous health benefits including improved cardiovascular health, stronger muscles, and better mental well-being. The WHO recommends at least 150 minutes of moderate physical activity per week.",
            "label": "low",
            "category": "distant_topic",
        },

        # === NO RELEVANCE: completely off-topic ===
        {
            "query": "What is the capital of France?",
            "passage": "The blue whale is the largest animal known to have ever existed. Adults can reach lengths of up to 30 meters and weigh as much as 173 tonnes.",
            "label": "none",
            "category": "irrelevant",
        },
        {
            "query": "How does photosynthesis work?",
            "passage": "The Great Wall of China stretches over 21,000 kilometers and was built over many centuries to protect Chinese states against nomadic invasions from the north.",
            "label": "none",
            "category": "irrelevant",
        },
        {
            "query": "What causes earthquakes?",
            "passage": "Italian cuisine is known for its regional diversity, especially between the north and south of the peninsula. Pasta, pizza, and olive oil are staples of the Italian diet.",
            "label": "none",
            "category": "irrelevant",
        },
        {
            "query": "How to sort a list in Python?",
            "passage": "The Olympic Games originated in ancient Greece around 776 BC. The modern Olympics were revived in 1896 by Pierre de Coubertin and have since become the world's foremost sports competition.",
            "label": "none",
            "category": "irrelevant",
        },
        {
            "query": "What is machine learning?",
            "passage": "Coffee is one of the most popular beverages in the world. It is brewed from roasted coffee beans, which are the seeds of berries from the Coffea plant native to tropical Africa.",
            "label": "none",
            "category": "irrelevant",
        },
        {
            "query": "Explain the theory of relativity",
            "passage": "Dogs have been domesticated for thousands of years and are valued for their companionship and loyalty. There are hundreds of breeds with varying sizes, shapes, and temperaments.",
            "label": "none",
            "category": "irrelevant",
        },

        # === EDGE CASES ===
        {
            "query": "Python",
            "passage": "Python is a high-level, general-purpose programming language created by Guido van Rossum. It emphasizes code readability with the use of significant indentation and supports multiple programming paradigms.",
            "label": "high",
            "category": "short_query",
        },
        {
            "query": "sort",
            "passage": "Sorting algorithms are methods for reordering elements in a list or array according to a comparison operator. Common algorithms include quicksort, mergesort, and heapsort, each with different time and space complexity trade-offs.",
            "label": "high",
            "category": "short_query",
        },
        {
            "query": "What is NOT a symptom of diabetes?",
            "passage": "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, slow-healing cuts and wounds, and unexplained weight loss.",
            "label": "medium",
            "category": "negation",
        },
        {
            "query": "What is the best programming language for web development?",
            "passage": "JavaScript is widely used for web development as it runs natively in browsers. Together with HTML and CSS, it forms the core technology stack for building interactive websites. Popular frameworks include React, Vue, and Angular for frontend development, while Node.js enables server-side JavaScript. Python with Django or Flask, Ruby with Rails, and PHP are also commonly used for backend web development. The choice of language depends on the project requirements, team expertise, and scalability needs.",
            "label": "high",
            "category": "long_passage",
        },
        {
            "query": "What is quantum computing and how does it differ from classical computing?",
            "passage": "Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can exist in multiple states simultaneously, potentially solving certain problems exponentially faster.",
            "label": "high",
            "category": "complex_query",
        },
        # Adversarial: keyword overlap but not answering
        {
            "query": "How to train a machine learning model?",
            "passage": "The machine was learning to operate at full capacity after the recent maintenance. The factory trained new workers on the model assembly line, and production increased significantly over the following months.",
            "label": "none",
            "category": "adversarial_keyword_overlap",
        },
        {
            "query": "What is the speed of light?",
            "passage": "Light beer has fewer calories than regular beer. The speed at which beer is consumed at sporting events has been the subject of several studies on public health and safety.",
            "label": "none",
            "category": "adversarial_keyword_overlap",
        },
    ]

    for i, pair in enumerate(pairs):
        pair["pair_id"] = i

    return pairs


if __name__ == "__main__":
    pairs = generate_test_pairs()
    print(f"Generated {len(pairs)} test pairs")
    for label in ["high", "medium", "low", "none"]:
        count = sum(1 for p in pairs if p["label"] == label)
        print(f"  {label}: {count}")
