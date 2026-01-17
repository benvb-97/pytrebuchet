Installation
============

From Source
~~~~~~~~~~~

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/benvb-97/pytrebuchet.git
   cd pytrebuchet

2. Install the package:

**Using pip:**

.. code-block:: bash

   # Basic installation
   pip install -e .

   # With development dependencies
   pip install -e ".[dev]"

**Using uv:**

.. code-block:: bash

   # Basic installation
   uv pip install -e .

   # With development dependencies
   uv sync --group dev
