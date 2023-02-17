from pymongo import MongoClient, database
from pymongo.errors import ConnectionFailure
from bson.objectid import ObjectId
import gridfs
from urllib.parse import quote_plus

class Store(object):
    _name = "store"
    current_client = None
    current_database = None

    def __repr__(self):
        return Store._name

    def __init__(self, mongo_host, mongo_port, mongo_user, mongo_password, database):
        if Store.current_client is None and Store.current_database is None:
            uri = "mongodb://%s:%s@%s" % (
                quote_plus(mongo_user), 
                quote_plus(mongo_password), 
                mongo_host + ":" + str(mongo_port)
            )
            try:
                Store.current_client = MongoClient(uri, connectTimeoutMS=5000, serverSelectionTimeoutMS=5000)
                Store.current_database = Store.current_client[database]

                # The ping command is cheap and does not require auth.
                Store.current_client.admin.command('ping')

            except ConnectionFailure:
                raise RuntimeError("Server not available")

        # connect to gridfs for maestro
        self.grid_fs = gridfs.GridFS(Store.current_database)

    def gridfs_insert(self, binary_data):
        return self.grid_fs.put(binary_data)
        
    def gridfs_get(self, _id):
        return self.grid_fs.get(_id).read()
    
    def _get_db(self, db=None):
        if db is None:
            return Store.current_database
        else:
            return Store.current_client[database]

    def get_stores(self, db=None):
        _db = self._get_db(db)
        return _db.list_collection_names()

    def get(self, store, query=None, db=None):
        _db = self._get_db(db)
        coll = _db.get_collection(store).find_one(query)
        if coll is not None:
            coll["_id"] = str(coll["_id"])
        return coll

    def get_by_id(self, store, _id, db=None):
        item = self.get(store, {"_id": ObjectId(_id)}, db=db)
        if item is not None:
            item["_id"] = str(item["_id"])
        return item

    def get_all(self, store, db=None):
        _db = self._get_db(db)
        coll = _db.get_collection(store).find()
        ret = []
        for c in coll:
            c["_id"] = str(c["_id"])
            ret.append(c)
        return ret

    def get_all_query(self, store, query, db=None):
        _db = self._get_db(db)
        coll = _db.get_collection(store).find(query)
        ret = []
        for c in coll:
            c["_id"] = str(c["_id"])
            ret.append(c)
        return ret

    def get_aggregation(self, store, pipeline, db=None):
        _db = self._get_db(db)
        coll = _db.get_collection(store).aggregate(pipeline)
        ret = []
        for c in coll:
            ret.append(c)
        return ret

    def exists(self, store, query, db=None):
        ret = self.get(store, query=query, db=db)
        if ret is not None:
            return True
        else:
            return False

    def clear_store(self, store, db=None):
        _db = self._get_db(db)
        _db.get_collection(store).drop()

    def find_and_modify(self, store, query, update, db=None):
        _db = self._get_db(db)
        return _db.get_collection(store).find_one_and_update(query, {"$set": update})

    def find_and_update(self, store, query, update, db=None):
        _db = self._get_db(db)
        return _db.get_collection(store).update_one(query, update)

    def update_one(self, store, query, update, db=None):
        _db = self._get_db(db)
        return _db.get_collection(store).update_one(query, {"$set": update})

    def update_many(self, store, query, update, db=None):
        _db = self._get_db(db)
        return _db.get_collection(store).update_many(query, {"$set": update})

    def delete_one(self, store, query, db=None):
        _db = self._get_db(db)
        return _db.get_collection(store).delete_one(query)

    def delete_many(self, store, query, db=None):
        _db = self._get_db(db)
        return _db.get_collection(store).delete_many(query)

    def insert_one(self, store, item, db=None):
        _db = self._get_db(db)
        i = _db.get_collection(store).insert_one(item)
        return str(i.inserted_id)

    def insert_many(self, store, items, db=None):
        _db = self._get_db(db)
        ids = _db.get_collection(store).insert_many(items)
        return [str(i) for i in ids.inserted_ids]

    def count(self, store, query=None, db=None):
        _db = self._get_db(db)
        return int(_db.get_collection(store).count_documents(query))