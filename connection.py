# tornado webserver framework
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import json
import Preprocessing


# Handler to handle recommend
class RequestHandler(tornado.web.RequestHandler):
    def initialize(self, password):
        self.password = password


    def post(self):
        db = Preprocessing.connect_database()
        c = db.cursor(Preprocessing.MySQLdb.cursors.DictCursor)
        c.callproc("getSetting", ["password"])
        password = c.fetchone()
        password = password['Value']
        c.close()

        try:
            data = json.loads(self.request.body)
            print(self.request.body)
            if data['retrain'] == '1':
                self.write("connected")
                Preprocessing.main()

            if data['retrain'] == '0':
                result = Preprocessing.prediction(data['news'])
                print(result)
                self.write(json.dumps({"result": result}))

        except Exception as e:
            self.set_status(500)


