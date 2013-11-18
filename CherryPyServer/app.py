import glob
import random
import time
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import cherrypy
from cherrypy.lib.static import serve_file
import threading
import texture_transfer

page1 = """
<!DOCTYPE html>
<html>
<head>
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css">

    <!-- Optional theme -->
    <link rel="stylesheet" href="//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-theme.min.css">

    <!-- Latest compiled and minified JavaScript -->
    <script src="//netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"></script>
    <title>Texture Transfer</title>
</head>
<body>
<div class="jumbotron">
     <div class="container">
        <h2>Texture Transfer Tool</h2>
        <p>Post your images here and result will appear in the gallery below in few minutes. <br/></p>
        <form class="form-horizontal" action="upload" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="texture">Texture:</label>
                <input type="file" id="texture" name="texture" required>
            </div>
            <div class="form-group">
                <label for="target">Target:</label>
                <input type="file" id="target" name="target" required>
            </div>
            <div class="form-group">
                <label for="block_size">Block Size:</label>
                <input type="number" name="block_size" min="10" max="100" value="36">
            </div>
            <div class="form-group">
            <input class="btn btn-primary btn-lg" type="submit" />
            </div>
        </form>
     </div>
</div>
<div class="container">
"""
page2 = """
</div>
</body>
</html>"""

def start_tt(source, target, blockW, blockH, num_of_iterations, overlap_x_frac, overlap_y_frac,
                 block_reduciton_factor, amount_to_probe_frac, output):

    t = texture_transfer.TextureTransferTool(source, target, blockW, blockH, num_of_iterations, overlap_x_frac, overlap_y_frac,
                 block_reduciton_factor, amount_to_probe_frac, 0)
    t.on_do_work(output)

def process_thread(source, target, block_size, accuracy, output):
    start_tt(source, target, int(block_size), int(block_size), 3, 0.8333, 0.33, 1/3.0, float(accuracy), output)
    os.remove(source)
    os.remove(target)

def latest_photos():
    path = os.path.join('data','results')
    files = glob.glob(os.path.join(path,'*.png'))
    files.sort()
    return files[-16:]

def populate_image_table():
    table = ""
    count = 0
    for i in reversed(latest_photos()):
        if count %4 == 0:
            table += '<div class="row"> '
        table += """
      <div class="col-sm-6 col-md-3">
        <a href="#" class="thumbnail">
          <img src=" images/""" + os.path.basename(i) + """ ">
        </a>
      </div>"""
        if count %4 == 3: table += '</div>'
        count +=1

    return table

class TextureTransfer(object):

    @cherrypy.expose
    def index(self):
        return page1 + populate_image_table() + page2

    @cherrypy.expose
    def upload(self, **data):
        process = time.time()
        os.makedirs(os.path.join(current_dir, 'process', str(process)))
        texture_file = os.path.join(current_dir,'process', str(process),'texture.jpg')
        target_file = os.path.join(current_dir,'process', str(process),'target.jpg')
        out_file = os.path.join(current_dir,'data','results', str(process))
        open(texture_file,'wb').write(data["texture"].file.read())
        open(target_file,'wb').write(data["target"].file.read())

        t = threading.Thread(target=process_thread, args=(texture_file, target_file, data['block_size'], 0.001, out_file))
        t.start()

        return "Your submission was received. Please wait..."

if __name__ == '__main__':
	import os
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	os.chdir(dname)
	if not os.path.exists('process'):
	    os.makedirs('process')
	conf = {'global':{
	    'server.socket_host': '0.0.0.0',
	    'server.socket_port': int(os.environ.get('PORT', '5001')),
	},
	    '/': {'tools.staticdir.root': os.path.join(current_dir, 'data'),},
	    '/images': { 'tools.staticdir.on': True,
	                'tools.staticdir.dir': 'results'

	    },
	    '/css': { 'tools.staticdir.on': True,
	                'tools.staticdir.dir': 'css'

	    },
	    '/js': { 'tools.staticdir.on': True,
	                'tools.staticdir.dir': 'js'

	    },
	    '/fonts': { 'tools.staticdir.on': True,
	                'tools.staticdir.dir': 'fonts'

	    }

	}
	cherrypy.quickstart(TextureTransfer(),'/', config= conf)

