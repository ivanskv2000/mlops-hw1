from flask_restx import Api

# from .controller import api as ns

api = Api(
    title="ML Models Api",
    version="1.0",
    description="MlOps course: Home Assignment #1 (Rest API)",
    contact="iaskvortsov@edu.hse.ru",
    doc="/ml_rest_api/doc",
)

# api.add_namespace(ns)
