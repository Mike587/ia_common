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
