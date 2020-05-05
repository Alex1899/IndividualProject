import JointAngles as ja
import pytest

parts1, parts2, parts3, parts_front = None, None, None, None


@pytest.fixture(scope='module')
def parts():
    global parts1, parts2, parts3, parts_front
    parts1 = [[[100, 8, 3], [200, 29, 6], [154, 9, 5], [20, 110, 12], [87, 330, 15], [500, 11, 18], [530, 220, 21]]]
    parts2 = [[[-100, -2, -3], [-300, -145, -6], [-530, 65, -9], [52, -11, -12], [-130, -70, -15], [-160, 42, -18],
               [75, -20, -21]]]
    parts_front = [[[567, 8, 3], [700, 29, 6], [-90, -2, -3], [-45, -178, -6], [645, 65, -6], [55, 9, 5], [200, 12, 10],
                    [52, -11, -12], [-130, -70, -15], [-160, 42, -18], [87, 330, 15], [500, 11, 18], [75, -20, -21],
                    [230, 54, 21]]]


def test_get_upper_arm_vectors(parts):
    assert ja.get_upper_arm_vectors(parts_front, view='front') == ([[657, 10]], [[745, 207]])
    assert ja.get_upper_arm_vectors(parts1) == [[-100, -21]]
    assert ja.get_upper_arm_vectors(parts2) == [[200, 143]]
    assert ja.get_upper_arm_vectors(parts3) is None


def test_get_forearm_vectors(parts):
    assert ja.get_forearm_vectors(parts_front, view='front') == ([[735, 67]], [[100, 187]])
    assert ja.get_forearm_vectors(parts1) == [[-46, -20]]
    assert ja.get_forearm_vectors(parts2) == [[-230, 210]]
    assert ja.get_forearm_vectors(parts3) is None


def test_get_trunk_vectors(parts):
    assert ja.get_trunk_vectors(parts_front, view='front') == [[148, 23]]
    assert ja.get_trunk_vectors(parts1) == [[-30, -209]]
    assert ja.get_trunk_vectors(parts2) == [[-235, 62]]
    assert ja.get_trunk_vectors(parts3) is None


def test_get_knee_vectors(parts):
    assert ja.get_knee_vects(parts1) == [[67, 220]]
    assert ja.get_knee_vects(parts2) == [[-182, -59]]
    assert ja.get_knee_vects(parts3) is None


def test_get_angles_functions():
    assert ja.get_upper_arm_trunk_angles([[-30, -209]], [[-100, -21]], [[200, 143]]) == ([133.73324661708065], [69.97174225500653])
    assert ja.get_upper_arm_trunk_angles([[-235, 62]], [[-100, -21]], [[200, 143]]) == ([129.6556718268453], [26.639339301067544])

    assert ja.get_upper_arm_forearm_angles([[-100, -21]], [[-46, -20]], [[200, 143]], [[-230, 210]]) == ([102.03779420946465], [11.638786555004181])
    assert ja.get_upper_arm_forearm_angles([[-100, -21]], [[-46, -20]], [[200, 143]]) is None

    assert ja.get_trunk_knee_angles([[-30, -209]], [[67, 220]], side='right') == [171.23060278563761]
    assert ja.get_trunk_knee_angles([[-30, -209]], [[67, 220]], side='left') == [188.76939721436239]
    assert ja.get_trunk_knee_angles([[-235, 62]], [[-182, -59]], side='right') == [32.74100094430668]
    assert ja.get_trunk_knee_angles([[-235, 62]], [[-182, -59]], side='left') == [327.2589990556933]
    assert ja.get_trunk_knee_angles([[-235, 62]], [[-182, -59]], side='front') is None
