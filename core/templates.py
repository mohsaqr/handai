"""
Handai Dataset Generation Templates
Pre-defined schemas for common dataset types
"""

DATASET_TEMPLATES = {
    "Custom (Define Your Own)": {
        "description": "Create your own dataset schema",
        "schema": {},
        "example_prompt": "Generate a row about..."
    },
    "Interview Questions & Answers": {
        "description": "Q&A pairs for training or evaluation",
        "schema": {
            "question": "str",
            "answer": "str",
            "category": "str",
            "difficulty": "str"
        },
        "example_prompt": "Generate an interview question and ideal answer about {topic}. Include category and difficulty level."
    },
    "Product Reviews": {
        "description": "Synthetic product reviews with sentiment",
        "schema": {
            "product_name": "str",
            "review_text": "str",
            "rating": "int",
            "sentiment": "str",
            "helpful_votes": "int"
        },
        "example_prompt": "Generate a realistic product review for {product_type}. Include rating 1-5 and sentiment analysis."
    },
    "Customer Support Tickets": {
        "description": "Support conversations for training",
        "schema": {
            "ticket_id": "str",
            "customer_message": "str",
            "category": "str",
            "priority": "str",
            "suggested_response": "str"
        },
        "example_prompt": "Generate a customer support ticket about {issue_type}. Include category, priority, and ideal response."
    },
    "Text Classification Dataset": {
        "description": "Labeled text samples for classification",
        "schema": {
            "text": "str",
            "label": "str",
            "confidence": "float"
        },
        "example_prompt": "Generate a text sample that belongs to the category '{category}'. The text should be realistic and clearly classifiable."
    },
    "Instruction-Response Pairs": {
        "description": "Training data for instruction-following",
        "schema": {
            "instruction": "str",
            "input": "str",
            "output": "str",
            "category": "str"
        },
        "example_prompt": "Generate an instruction-following example for {task_type}. Include clear instruction, optional input, and ideal output."
    },
    "Code Snippets": {
        "description": "Code examples with explanations",
        "schema": {
            "language": "str",
            "code": "str",
            "explanation": "str",
            "complexity": "str",
            "use_case": "str"
        },
        "example_prompt": "Generate a {language} code snippet that demonstrates {concept}. Include explanation and use case."
    },
    "Named Entity Recognition": {
        "description": "Text with entity annotations",
        "schema": {
            "text": "str",
            "entities": "list",
            "entity_types": "list"
        },
        "example_prompt": "Generate a sentence containing named entities (persons, organizations, locations). List the entities found."
    },
    "Sentiment Analysis": {
        "description": "Text samples with sentiment labels",
        "schema": {
            "text": "str",
            "sentiment": "str",
            "confidence": "float",
            "aspects": "list"
        },
        "example_prompt": "Generate a text expressing {sentiment} sentiment about {topic}. Include aspect-level sentiment if applicable."
    },
    "Question Answering": {
        "description": "Context-question-answer triplets",
        "schema": {
            "context": "str",
            "question": "str",
            "answer": "str",
            "answer_start": "int"
        },
        "example_prompt": "Generate a context paragraph about {topic}, a question about it, and the exact answer from the context."
    },
}


def get_template_names():
    """Get list of all template names"""
    return list(DATASET_TEMPLATES.keys())


def get_template(name: str) -> dict:
    """Get a template by name"""
    return DATASET_TEMPLATES.get(name, DATASET_TEMPLATES["Custom (Define Your Own)"])


def get_template_schema(name: str) -> dict:
    """Get just the schema from a template"""
    template = get_template(name)
    return template.get("schema", {})
