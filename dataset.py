import pymysql
import json

HOST = '192.168.41.46'
DATABASE = 'web3'
USER = 'web3'
PASSWORD = 'web3'
PORT = 3306
CHARSET = 'utf8'
TYPE = {
    'fee': 0,
    'disableTrading': 1,
    'blacklist': 2,
    'reflect': 3,
    'maxTX': 4,
    'mint': 5,
    'honeypot': 6,
    'reward': 7,
    'rebase': 8,
    'maxSell': 9
}

db = pymysql.connect(
    host=HOST,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
    port=PORT,
    charset=CHARSET
)

cursor = db.cursor()


def getXY(id):
    sql = "SELECT c.Id, c.ContractAddress, c.TokenIds, t.Scams FROM contracts AS c INNER JOIN tokens AS t on c.ContractAddress=t.ContractAddress WHERE c.Id=%d" % id
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
        tokenIds = json.loads(res[0][2])
        x = []
        for i in tokenIds:
            for j in tokenIds[i]:
                x.append(tokenIds[i][j])
        scams = json.loads(res[0][3])
        y = [0] * 10
        for item in scams:
            y[TYPE[item['type']]] = 1
        return {'x': x, 'y': y}
    except Exception as e:
        return None


# for my model with pre-trained smartbert embeddings
def getXY2(id):
    sql = "SELECT c.Id, c.ContractAddress, c.Embedding, t.Scams, c.EmbeddingMax FROM contracts AS c INNER JOIN tokens AS t on c.ContractAddress=t.ContractAddress WHERE c.Id=%d" % id
    try:
        cursor.execute(sql)
        res = cursor.fetchall()

        # average embedding
        data = json.loads(res[0][2])
        x = []
        for i in data:
            for j in data[i]:
                x.append(data[i][j])
        # max embedding
        data = json.loads(res[0][4])
        x2 = []
        for i in data:
            for j in data[i]:
                x2.append(data[i][j])
        # y
        scams = json.loads(res[0][3])
        y = [0] * 10
        for item in scams:
            y[TYPE[item['type']]] = 1
        return {'x': x, 'x2': x2, 'y': y}
    except Exception as e:
        return None


# res = getXY2(140)
# print(res)
