import bench


def test_tool_arguments_of_the_wrong_shape_fail_the_call_instead_of_crashing_the_grader():
    for raw in ('[1]', '"x"', '3', 'true'):
        r = {"tool_calls": [{"function": {"name": "get_weather", "arguments": raw}}]}
        assert bench.parse_calls(r) == [("get_weather", None)]
        assert not bench.t_single(r)[0]
