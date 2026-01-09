"""
Basic usage example for SPARK
"""
from spark.core import hello_world, Example


def main():
    """Demonstrate basic functionality"""
    # Simple function call
    print(hello_world())

    # Class usage
    example = Example("User")
    print(example.greet())


if __name__ == "__main__":
    main()
