from flask import Flask, send_from_directory, render_template
import os

app = Flask(__name__)

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('output', path)

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')


@app.route('/')
def index():
    # Get list of images in the output folder
    images = os.listdir('output')
    # Render the template with the images
    return render_template('index.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
