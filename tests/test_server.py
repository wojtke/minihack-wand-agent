"""Real native games, session exhaustion/recovery and browser request validation."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from wand_agent import serve


class GameServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), serve.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}/api/'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def tearDown(self):
        with serve.ENGINE_LOCK:
            for game in serve.GAMES.values(): game.close()
            serve.GAMES.clear()

    def post(self, path, data, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin: headers['Origin'] = origin
        with urlopen(Request(self.base+path, json.dumps(data).encode(), headers), timeout=20) as r:
            return json.load(r)

    def test_abandoned_tabs_cannot_block_new_games(self):
        sessions = [self.post('reset', {'level': 'Easy', 'seed': i})['session']
                    for i in range(serve.MAX_GAMES+2)]
        self.assertEqual(len(serve.GAMES), serve.MAX_GAMES)
        with self.assertRaises(HTTPError) as error:
            self.post('step', {'session': sessions[0], 'key': 'z'})
        self.assertEqual(error.exception.code, 404)
        state = self.post('step', {'session': sessions[-1], 'key': 'z'})
        self.assertIn('What do you want to zap?', state['message'])
        self.assertTrue(self.post('close', {'session': sessions[-1]})['closed'])
        self.assertNotIn(sessions[-1], serve.GAMES)
        self.assertTrue(self.post('close', {'session': sessions[-1]})['closed'])

    def test_native_success_and_same_seed_restart(self):
        initial = self.post('reset', {'level': 'Easy', 'seed': 0})
        state = initial
        for key in 'zfl': state = self.post('step', {'session': state['session'], 'key': key})
        self.assertEqual(state['status'], 'TASK_SUCCESSFUL')
        self.assertAlmostEqual(state['total'], .98)
        restarted = self.post('reset', {'level': 'Easy', 'seed': 0, 'session': state['session']})
        self.assertEqual(restarted['image'], initial['image'])
        self.assertEqual(restarted['steps'], 0)
        self.assertEqual(len(serve.GAMES), 1)

    def test_invalid_requests(self):
        for path, data, origin, code in [
            ('reset', {'session': []}, None, 400),
            ('reset', {'level': 'Fake'}, None, 400),
            ('step', {'key': 'x'}, None, 400),
            ('reset', {}, 'https://example.com', 403),
        ]:
            with self.assertRaises(HTTPError) as error: self.post(path, data, origin)
            self.assertEqual(error.exception.code, code)


if __name__ == '__main__': unittest.main()
