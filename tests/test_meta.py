import numpy as np
import pytest

from ia_common.czi import meta

XML = """<ImageDocument><Metadata>
  <Scaling><Items>
    <Distance Id="X"><Value>2.0568685192229183E-07</Value></Distance>
    <Distance Id="Y"><Value>2.0568685192229183E-07</Value></Distance>
    <Distance Id="Z"><Value>2E-06</Value></Distance>
  </Items></Scaling>
  <Information><Image>
    <SizeZ>{nz}</SizeZ>
    <Dimensions>
      {z}
      <Channels><Channel Id="Channel:0" Name="DAPI"/><Channel Id="Channel:1" Name="DAPI#"/></Channels>
      <S><Scenes><Scene Index="0" Name="A1"><CenterPosition>7300,19100</CenterPosition></Scene></Scenes></S>
    </Dimensions>
  </Image></Information>
  <ParameterCollection Id="MTBStageAxisX"><Position>6994.5</Position></ParameterCollection>
  <ParameterCollection Id="MTBStageAxisY"><Position>18916.5</Position></ParameterCollection>
  <ParameterCollection Id="MTBFocus"><Position>1383.0</Position></ParameterCollection>
</Metadata></ImageDocument>"""
Z_OK = "<Z><Positions><Interval><Start>-19</Start><Increment>2</Increment></Interval></Positions></Z>"


def root(nz=20, z=Z_OK):
    return meta.parse(XML.format(nz=nz, z=z))


def test_basic_fields():
    r = root()
    assert meta.pixel_size_m(r) == pytest.approx(0.20568685e-6)
    assert meta.channel_names(r) == ["DAPI", "DAPI#"]
    assert meta.scene_names(r) == {0: "A1"}
    assert meta.planned_scene_centres_m(r)[0] == pytest.approx((7.3e-3, 19.1e-3))
    assert meta.recorded_stage_xy_m(r) == pytest.approx((6994.5e-6, 18916.5e-6))
    assert meta.recorded_focus_m(r) == pytest.approx(1383.0e-6)


def test_z_relative():
    z = meta.z_relative_m(root())
    assert len(z) == 20 and z[0] == pytest.approx(-19e-6) and z[-1] == pytest.approx(19e-6)
    assert np.mean(z) == pytest.approx(0.0, abs=1e-12)
    assert meta.z_step_m(root()) == pytest.approx(2e-6)


def test_z_missing_spacing_raises_instead_of_guessing():
    with pytest.raises(meta.MetadataError):
        meta.z_relative_m(root(z=""))
    # a single plane needs no spacing
    assert meta.z_relative_m(root(nz=1, z="")).tolist() == [0.0]


def test_missing_or_nonsquare_scaling_raises():
    with pytest.raises(meta.MetadataError):
        meta.pixel_size_m(meta.parse("<ImageDocument/>"))
    bad = XML.format(nz=1, z="").replace('<Distance Id="Y"><Value>2.0568685192229183E-07',
                                           '<Distance Id="Y"><Value>3E-07')
    with pytest.raises(meta.MetadataError):
        meta.pixel_size_m(meta.parse(bad))


# --- sample_carrier ------------------------------------------------------
# Trimmed from a Marc_SM_002 .czexp: an insert slide (the file records only the slide).
INSERT_XML = """<ImageDocument><Metadata><Experiment><SampleHolder>
    <Template Name="Chamber Slide 8x - 76 x 26 mm - Vertical">
      <AnchorPoint>0,0</AnchorPoint><BoundsSize>26000,76000</BoundsSize>
      <Category>Multichamber</Category><ShapeType>Rectangle</ShapeType>
      <ShapeColumns>2</ShapeColumns><ShapeRows>4</ShapeRows>
      <ShapeWidth>9500</ShapeWidth><ShapeHeight>10600</ShapeHeight>
      <ShapeDistanceX>10900</ShapeDistanceX><ShapeDistanceY>12100</ShapeDistanceY>
      <BottomMaterial>Glass</BottomMaterial><BottomThickness>186</BottomThickness>
      <BottomOffsetMeasured>NaN</BottomOffsetMeasured>
      <InsertType>PartOfInsert</InsertType><PartOfInsertName></PartOfInsertName>
    </Template></SampleHolder></Experiment></Metadata></ImageDocument>"""


def test_sample_carrier_insert_records_only_the_slide():
    c = meta.sample_carrier(meta.parse(INSERT_XML))
    assert c["name"] == "Chamber Slide 8x - 76 x 26 mm - Vertical"
    assert (c["rows"], c["columns"]) == (4, 2)
    assert c["well_width_m"] == pytest.approx(9.5e-3)           # µm in the file → m
    assert c["pitch_y_m"] == pytest.approx(12.1e-3)
    assert c["bounds_size_m"] == pytest.approx((0.026, 0.076))
    assert c["bottom_thickness_m"] == pytest.approx(186e-6)
    assert c["is_part_of_insert"] is True and c["part_of_insert_name"] is None


def test_sample_carrier_falls_back_to_czsht_copy():
    xml = """<X><ScanAreaSampleHolderInfo><Template Name="Multiwell 96.czsht">
      <ShapeRows>8</ShapeRows></Template></ScanAreaSampleHolderInfo></X>"""
    c = meta.sample_carrier(meta.parse(xml))
    assert c["name"] == "Multiwell 96" and c["rows"] == 8
    assert c["pitch_x_m"] is None                               # missing tag → None


def test_sample_carrier_conflict_warns_and_prefers_sampleholder(caplog):
    xml = """<X><SampleHolder><Template Name="Multiwell 96" /></SampleHolder>
      <ScanAreaSampleHolderInfo><Template Name="Multichamber 384.czsht" /></ScanAreaSampleHolderInfo></X>"""
    c = meta.sample_carrier(meta.parse(xml))
    assert c["name"] == "Multiwell 96"
    assert "more than one sample carrier" in caplog.text


def test_sample_carrier_none_without_template():
    assert meta.sample_carrier(meta.parse("<X><SampleHolder/></X>")) is None


def test_sample_carrier_from_a_real_czi(data):
    from pylibCZIrw import czi as pyczi
    with pyczi.open_czi(str(data / "WF_PartialChip_off_centered.czi")) as d:
        c = meta.sample_carrier(meta.parse(d.raw_metadata))
    assert c["name"] == "Slide 76 x 26 mm - Vertical"
