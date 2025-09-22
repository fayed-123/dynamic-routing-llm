# app.py (النسخة المحدثة النهائية)

from flask import Flask, render_template, request, jsonify
from routing_system import create_routing_system
from evaluator import SystemEvaluator
from test_queries import quick_test_queries

app = Flask(__name__)

# تهيئة المكونات الرئيسية مرة واحدة
print("🚀 Initializing System Components for the web app...")
routing_system = create_routing_system()
evaluator = SystemEvaluator()
print("✅ System is ready.")

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    يعالج الصفحة الرئيسية وطلبات الاستعلام العادية.
    """
    if request.method == 'POST':
        query = request.form.get('query')
        if query and query.strip():
            result = routing_system.process_query(query)
            # عند إرسال استعلام، نعرض فقط نتيجته
            return render_template('index.html', query=query, result=result)

    # عند فتح الصفحة أول مرة، لا نعرض أي نتائج
    return render_template('index.html')


@app.route('/run_evaluation')
def run_evaluation():
    """
    يشغل التقييم السريع ويعرض ملخص النتائج.
    """
    print("🔍 Starting quick evaluation from web interface...")
    test_queries = quick_test_queries()
    evaluation_summary = evaluator.evaluate_system(test_queries)
    print("✅ Evaluation finished.")
    # عند تشغيل التقييم، نعرض فقط ملخص التقييم
    return render_template('index.html', evaluation_summary=evaluation_summary)


# --- الكود الجديد والمهم يبدأ هنا ---
@app.route('/show/<info_type>')
def show_info(info_type):
    """
    نقطة نهاية ديناميكية لعرض المعلومات الإدارية (stats, cache, help).
    """
    if info_type == 'stats':
        stats_data = routing_system.get_detailed_stats()
        return render_template('index.html', stats_data=stats_data)

    elif info_type == 'cache':
        cache_data = routing_system.cache_manager.get_cache_info()
        return render_template('index.html', cache_data=cache_data)

    elif info_type == 'help':
        help_data = {
            "Available Commands": {
                "Process Query": "Enter any text in the main box and click 'Process Query'.",
                "Run Quick Evaluation": "Runs a pre-defined set of test queries and shows a performance summary.",
                "Show Stats": "Displays live statistics about total queries, success rate, and model usage.",
                "Show Cache Info": "Shows the current state of the cache, including the most accessed items."
            }
        }
        return render_template('index.html', help_data=help_data)

    # إذا كان الرابط غير معروف، ارجع للصفحة الرئيسية
    return render_template('index.html')
# ------------------------------------


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/stats')
def api_stats():
    if routing_system:
        stats = routing_system.get_detailed_stats()
        return jsonify(stats)
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)