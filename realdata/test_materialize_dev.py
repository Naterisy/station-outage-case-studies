import hashlib
import unittest
import numpy as np
from materialize_dev import BASE, checked, daily, source

class Materializer(unittest.TestCase):
    def test_locked_year_rejected_even_with_metadata(self):
        with self.assertRaises(ValueError):
            source({}, 2023, "01234567890")

    def test_changed_snapshot_rejected(self):
        with self.assertRaises(ValueError):
            checked(b"b", {"bytes":1,"sha256":hashlib.sha256(b"a").hexdigest()})

    def test_source_host_frozen(self):
        station="01234567890"
        row={"url":f"{BASE}/2021/{station}.csv","bytes":1,"sha256":"e"*64}
        self.assertEqual(source({(2021,station):row},2021,station)[0],row["url"])
        row["url"]="https://example.org/data.csv"
        with self.assertRaises(ValueError):
            source({(2021,station):row},2021,station)

    def test_masks_sentinel_and_low_count(self):
        data=("STATION,DATE,TEMP,TEMP_ATTRIBUTES\n"
              "01234567890,2021-01-01,50,4\n"
              "01234567890,2021-01-02,50,3\n"
              "01234567890,2021-01-04,9999.9,24\n").encode()
        x=daily(data,"01234567890",2021)
        self.assertEqual(x[0],10.)
        self.assertTrue(np.isnan(x[1:]).all())
