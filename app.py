# app.py

import sys
from flask import Flask, render_template, request, jsonify
from routing_system import create_routing_system
from evaluator import SystemEvaluator
from test_queries import quick_test_queries

# Ensure console supports UTF-8 for logging
if sys.platform == "win32":
    try:
        import os
        os.system("chcp 65001 > nul")
    except Exception:
        pass

app = Flask(__name__)

# Initialize system components once on startup
print("🚀 Initializing System Components...")
routing_system = create_routing_system()
evaluator = SystemEvaluator()
print("✅ System is ready.")

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main page for submitting queries."""
    if request.method == 'POST':
        query = request.form.get('query')
        if query and query.strip():
            result = routing_system.process_query(query)
            return render_template('index.html', query=query, result=result)
    return render_template('index.html')

@app.route('/run_evaluation')
def run_evaluation():
    """Runs a quick evaluation and displays the summary."""
    print("🔍 Starting quick evaluation from web interface...")
    test_queries = quick_test_queries()
    evaluation_summary = evaluator.evaluate_system(test_queries)
    print("✅ Evaluation finished.")
    return render_template('index.html', evaluation_summary=evaluation_summary)

@app.route('/show/<info_type>')
def show_info(info_type):
    """Dynamically displays administrative info (stats, cache, help)."""
    if info_type == 'stats':
        return render_template('index.html', stats_data=routing_system.get_detailed_stats())
    elif info_type == 'cache':
        return render_template('index.html', cache_data=routing_system.cache_manager.get_cache_info())
    elif info_type == 'help':
        help_data = {
            "Available Actions": {
                "Process Query": "Submit text to the routing system.",
                "Run Quick Evaluation": "Test the system on a small, predefined set of queries.",
                "Show Stats": "View real-time statistics on system performance.",
                "Show Cache Info": "Inspect the contents of the query cache.",
                "Dashboard": "View a live, auto-updating dashboard of system metrics."
            }
        }
        return render_template('index.html', help_data=help_data)
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Serves the live dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint to provide live stats for the dashboard."""
    if routing_system:
        return jsonify(routing_system.get_detailed_stats())
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')