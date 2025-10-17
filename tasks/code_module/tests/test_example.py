from code_module import example


def test_add():
    assert example.add(2, 3) == 5


def test_multiply_exists():
    assert hasattr(example, "multiply")
