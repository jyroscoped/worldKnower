"""
Unit tests for the pure-Python helper functions in worldknower.py.

Run with:  python -m pytest test_worldknower.py -v
"""

import math
import pytest

from worldknower import description_to_latlon, latlon_to_minimap_pixel, REGION_COORDS


# ---------------------------------------------------------------------------
# description_to_latlon
# ---------------------------------------------------------------------------

class TestDescriptionToLatLon:
    def test_exact_country_name(self):
        lat, lon = description_to_latlon("This is france")
        assert lat == pytest.approx(46.23, abs=0.1)
        assert lon == pytest.approx(2.21, abs=0.1)

    def test_case_insensitive(self):
        lat, lon = description_to_latlon("A street in JAPAN")
        assert lat == pytest.approx(36.20, abs=0.1)
        assert lon == pytest.approx(138.25, abs=0.1)

    def test_adjective_form(self):
        lat, lon = description_to_latlon("russian highway in winter")
        assert lat == pytest.approx(61.52, abs=0.1)
        assert lon == pytest.approx(105.32, abs=0.1)

    def test_continent_fallback(self):
        lat, lon = description_to_latlon("somewhere in europe")
        assert lat == pytest.approx(54.53, abs=0.1)

    def test_longer_phrase_wins_over_shorter(self):
        # "united states" should win over "states" being absent,
        # and "south america" should win over "america"
        lat, lon = description_to_latlon("somewhere in south america")
        expected_lat, expected_lon = REGION_COORDS["south america"]
        assert lat == pytest.approx(expected_lat, abs=0.1)
        assert lon == pytest.approx(expected_lon, abs=0.1)

    def test_unknown_returns_default(self):
        lat, lon = description_to_latlon("a mysterious landscape")
        assert lat == pytest.approx(20.0, abs=0.1)
        assert lon == pytest.approx(0.0, abs=0.1)

    def test_all_regions_present(self):
        """Every entry in REGION_COORDS must round-trip through the lookup."""
        for region, (expected_lat, expected_lon) in REGION_COORDS.items():
            lat, lon = description_to_latlon(f"This photo was taken in {region}")
            assert lat == pytest.approx(expected_lat, abs=0.1), region
            assert lon == pytest.approx(expected_lon, abs=0.1), region


# ---------------------------------------------------------------------------
# latlon_to_minimap_pixel
# ---------------------------------------------------------------------------

WORLD_BOUNDS = {"north": 85.0, "south": -85.0, "west": -180.0, "east": 180.0}


class TestLatLonToMinimapPixel:
    def test_centre_of_map(self):
        """(0, 0) should land near the centre-left of a 1000×600 map."""
        px, py = latlon_to_minimap_pixel(0.0, 0.0, WORLD_BOUNDS, 1000, 600)
        # longitude 0 → x = 500
        assert px == pytest.approx(500, abs=2)
        # latitude 0 → y is near middle (Mercator stretches high latitudes)
        assert 250 < py < 350

    def test_top_left_corner(self):
        """North-West corner should map to (0, 0) pixel."""
        px, py = latlon_to_minimap_pixel(85.0, -180.0, WORLD_BOUNDS, 1000, 600)
        assert px == 0
        assert py == 0

    def test_bottom_right_corner(self):
        """South-East corner should map to (width, height) pixel."""
        px, py = latlon_to_minimap_pixel(-85.0, 180.0, WORLD_BOUNDS, 1000, 600)
        assert px == 1000
        assert py == 600

    def test_usa_is_left_of_centre(self):
        """USA centre (~39.5°N, -98.35°E) should be in the left half of the map."""
        px, py = latlon_to_minimap_pixel(39.5, -98.35, WORLD_BOUNDS, 1000, 600)
        assert px < 500

    def test_australia_is_right_and_low(self):
        """Australia (~-25°S, 133°E) should be right-of-centre and below middle."""
        px, py = latlon_to_minimap_pixel(-25.27, 133.78, WORLD_BOUNDS, 1000, 600)
        assert px > 500
        assert py > 300

    def test_output_clamped_to_map(self):
        """Out-of-range coordinates should be clamped to the map boundaries."""
        px, py = latlon_to_minimap_pixel(90.0, 200.0, WORLD_BOUNDS, 800, 500)
        assert 0 <= px <= 800
        assert 0 <= py <= 500

    def test_mercator_northern_stretch(self):
        """
        In Web-Mercator, high latitudes are *stretched* (more pixels per degree
        near the poles), so the 30° band from 60°N→30°N occupies more vertical
        pixels than the equatorial 30° band from 30°N→0°.
        """
        _, y_60 = latlon_to_minimap_pixel(60.0, 0.0, WORLD_BOUNDS, 1000, 1000)
        _, y_30 = latlon_to_minimap_pixel(30.0, 0.0, WORLD_BOUNDS, 1000, 1000)
        _, y_0 = latlon_to_minimap_pixel(0.0, 0.0, WORLD_BOUNDS, 1000, 1000)

        # 60°→30° band (near pole) must span MORE pixels than 30°→0° band.
        assert (y_30 - y_60) > (y_0 - y_30)
