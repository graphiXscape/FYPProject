from flask import Blueprint, request, jsonify
from appbackend.services.logo_service import register_logo_service, lookup_logo_service

logo_bp = Blueprint('logo', __name__)

@logo_bp.route('/register-logo', methods=['POST'])
def register_logo():
    return register_logo_service(request)

@logo_bp.route('/lookup-logo', methods=['POST'])
def lookup_logo():
    return lookup_logo_service(request)