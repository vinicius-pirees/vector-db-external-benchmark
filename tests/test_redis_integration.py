import unittest
import os
import shutil
from vector_db_external.vectordb.redis import Redis

os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PASSWORD"] = "s1234"
os.environ["REDIS_PORT"] = "8002"


class TestRedisClient(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up any necessary configurations or mocks here
        client = Redis(database_name="test_db", vector_dimension=3)
        client.remove_index()
    @classmethod
    def tearDownClass(self):
        # Clean up any resources used by the tests
        ...

    def test_insert_embeddings(self):
        # Create an instance of ChromaClient
        client = Redis(database_name="test_db", vector_dimension=3)

        # Mock data
        ids = ["doc1", "doc2", "doc3"]
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [3.0, 5.0, 6.0]]
        documents = ["text1", "text2", ""]
        metadata = [{"key": "value", "key2": "value2"}, {"key": "value"}, {"key": "val"}]

        # Call insert_embeddings
        try:
            client.insert_embeddings(ids=ids, embeddings=embeddings, documents=documents, metadata=metadata)
        except Exception as e:
            # If an exception occurs, fail the test with an informative message
            self.fail(f"Unexpected exception: {e}")

        # If no exception occurs, the test passes
        self.assertTrue(True)

   
    
    def test_insert_embeddings_without_documents(self):
        # Create an instance of ChromaClient
        client = Redis(database_name="test_db", vector_dimension=3)

        # Mock data
        ids = ["doc4"]
        embeddings = [[700.0, 800.0, 300.0]]

        # Call insert_embeddings without passing documents
        try:
            client.insert_embeddings(ids=ids, embeddings=embeddings)
        except Exception as e:
            # If an exception occurs, fail the test with an informative message
            self.fail(f"Unexpected exception: {e}")

        # If no exception occurs, the test passes
        self.assertTrue(True)
    

    def test_search_embedding(self):
        # Create an instance of ChromaClient
        client = Redis(database_name="test_db", vector_dimension=3)

        # Mock data
        query_embedding = [1.0, 2.0, 3.0]
        k = 2

        # Call search_embedding
        result = client.search_embedding(query=query_embedding, k=k)

        self.assertEqual(len(result.ids), 2)

        self.assertEqual(result.ids[0], "doc1")
        self.assertEqual(result.documents[0], "text1")

        # the next closest vector is [3.0, 5.0, 6.0]
        self.assertEqual(result.ids[1], "doc3") 
        self.assertEqual(result.documents[1], "")

    def test_search_embedding_with_filter(self):
        # Create an instance of ChromaClient
        client = Redis(database_name="test_db", vector_dimension=3)

        # Mock data
        query_embedding = [1.0, 2.0, 3.0]
        k = 2
        filters = {"key": "value"}

        # Call search_embedding
        result = client.search_embedding(query=query_embedding, k=k, filters=filters)

        self.assertEqual(len(result.ids), 2)

        self.assertEqual(result.ids[0], "doc1")
        self.assertEqual(result.documents[0], "text1")

        # when filtered, the next closest vector is [4.0, 5.0, 6.0]
        self.assertEqual(result.ids[1], "doc2")
        self.assertEqual(result.documents[1], "text2")


    ##TODO: test 2 filters

    
    def test_search_embedding_no_docs(self):
        client = Redis(database_name="test_db", vector_dimension=3)

        # Mock data
        query_embedding = [700.0, 800.0, 300.0]
        k = 1

        # Call search_embedding
        result = client.search_embedding(query=query_embedding, k=k)

        self.assertEqual(len(result.ids), 1)
        self.assertEqual(result.ids[0], "doc4")
        self.assertEqual(result.documents[0], None)
    

if __name__ == '__main__':
    unittest.main()
