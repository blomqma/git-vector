def test_version(mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-api-key"})

    from git_vector import __version__

    assert __version__ == "0.1.0-rc1"
