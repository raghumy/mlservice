#!/usr/bin/env python3
import connexion
import datetime
import logging

from connexion.resolver import RestyResolver

from connexion import NoContent

from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
#logging.getLogger('flask_cors').level = logging.DEBUG

app = connexion.App(__name__)

app.add_api('swagger.yaml', resolver=RestyResolver('api'))

cors = CORS(app.app)

if __name__ == '__main__':
    app.run(port=8080)
