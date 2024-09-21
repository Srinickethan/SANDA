# SANDA 

This repository contains a FastAPI application for making predictions using a pre-trained model.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Srinickethan/SANDA.git
    cd SANDA
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI application:
    ```sh
    uvicorn src.main:app --reload
    ```

2. The application will be available at `http://127.0.0.1:8000`.

## Endpoints

- `GET /predict/`: Make a prediction based on the input text.

## Example

To make a prediction, send a GET request to the `/predict/` endpoint with a JSON body containing the input text. For example:

```sh
curl -X 'GET' \
  'http://localhost:8000/predict/?text=This%20is%20a%20test%20sentence.' \
    -H 'accept: application/json'
```

The response will be a JSON object containing the cluster ID.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
