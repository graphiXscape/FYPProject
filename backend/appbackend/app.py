from flask import Flask
from appbackend.config import Config
from appbackend.extensions import cors
from appbackend.routes.logo import logo_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    cors.init_app(app)
    app.register_blueprint(logo_bp, url_prefix='/api')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(port=5000, debug=True, use_reloader=False)