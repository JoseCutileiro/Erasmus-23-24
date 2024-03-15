from flask import Flask, jsonify
import random
import os

app = Flask(__name__)


# Route to get all books
@app.route('/map_info/<int:room_id>', methods=['GET'])
def get_map_info(room_id):
    rx = room_id % 100
    ry = (room_id + 37) % 100
    rdiv_x = (room_id * 79) % 10 + 1
    rdiv_y = ((room_id * 16) % 10 + (room_id * 83 % 10))%10 + 1
    if (ry < 30):
        ry += 20
    
    ret = [rx,ry,rdiv_x,rdiv_y]
    return jsonify(ret)


# Route to get all books
@app.route('/join_room/<int:room_id>/<int:player_id>', methods=['GET'])
def join_room(room_id,player_id):
    file_name = f"{player_id}.room"
    directory = f"room/{room_id}/"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)
    open(file_path, 'a').close()
    
    
    return "Sucess"


if __name__ == '__main__':
    app.run(debug=True)