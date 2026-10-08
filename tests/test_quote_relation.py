from apv_rag.quote_relation import token_features,decode_speaker

def test_features_do_not_include_annotated_speaker_or_quote_direction():
    tokens=['Alice','said','hello','world']
    features=token_features(tokens,(2,4))
    assert features[0]['quote_side']=='before'
    assert features[1]['next1']=='hello'
    assert all('label' not in x and 'speaker' not in x for x in features)

def test_decoder_excludes_quote_and_preserves_multitoken_name():
    assert decode_speaker(['Mary','Jones','said','hello'],(3,4),[.9,.8,.1,1],.5)=='Mary Jones'
    assert decode_speaker(['Mary','said','hello'],(2,3),[.1,.1,1],.5) is None
