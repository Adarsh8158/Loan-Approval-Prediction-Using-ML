"""
Flask Application for Loan Approval Prediction & Admin Dashboard
Serves the dark neo-glassmorphic Bank Admin Dashboard, provides
real-time machine learning prediction APIs, and manages SQLite audit history.
"""

from pathlib import Path
from flask import Flask, render_template, request, jsonify
from src.predictor import predict_loan
from src.database import save_prediction_record, get_prediction_history, clear_prediction_history

# Project base directory
BASE_DIR = Path(__file__).resolve().parent

# Initialize Flask with explicit templates and static directories
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)


@app.route("/", methods=["GET", "POST"])
def dashboard():
    """Render the primary Bank Admin Dashboard."""
    prediction_result = None
    error_message = None

    if request.method == "POST":
        try:
            form_payload = request.form.to_dict()
            prediction_result = predict_loan(form_payload)
            save_prediction_record(form_payload, prediction_result)
        except Exception as e:
            error_message = f"Error processing prediction: {str(e)}"

    recent_history = get_prediction_history(limit=10)

    return render_template(
        "index.html",
        prediction=prediction_result,
        error=error_message,
        history=recent_history
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for real-time AJAX loan approval evaluation.
    Used by the dashboard prediction modal. Saves to SQLite.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        result = predict_loan(data)
        save_prediction_record(data, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/history", methods=["GET", "DELETE"])
def api_history():
    """Retrieve or clear prediction history."""
    if request.method == "DELETE":
        clear_prediction_history()
        return jsonify({"message": "History cleared successfully"})
    history = get_prediction_history(limit=50)
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
