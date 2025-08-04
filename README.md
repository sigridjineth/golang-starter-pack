## Project Structure

The project follows the standard Go project layout:

- `/cmd`: Main application entrypoints.
- `/internal`: Private application and library code.
  - `/application`: Business logic (services).
  - `/domain`: Domain models and interfaces.
  - `/infrastructure`: Data access layer (repositories).
  - `/interfaces`: Adapters to external systems (e.g., HTTP, gRPC).
- `/pkg`: Public library code.
- `/api`: API definitions (e.g., OpenAPI, protobuf).
- `/configs`: Configuration files.
- `/scripts`: Build and automation scripts.
- `/test`: Integration and end-to-end tests.

## Getting Started

1.  **Copy configuration:**
    ```bash
    cp configs/config.yaml.example configs/config.yaml
    ```
2.  **Install dependencies:**
    ```bash
    make tidy
    ```
3.  **Run the application:**
    ```bash
    make run
    ```

## Available Commands

- `make run`: Run the main application.
- `make build`: Build the application binary.
- `make test`: Run unit tests.
- `make cover`: Run unit tests with coverage report.
- `make lint`: Run the linter.
- `make tidy`: Tidy go modules.
