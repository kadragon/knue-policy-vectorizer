import pytest

from src.knue_board_ingestor import KnueBoardIngestor


@pytest.mark.unit
def test_parse_rss_basic_with_description_html_stripped():
    xml = """
    <rss version="2.0">
      <channel>
        <title>Example Feed</title>
        <item>
          <title><![CDATA[ Post <b>One</b> ]]></title>
          <link>https://example.com/post/1</link>
          <pubDate>Sun, 17 Aug 2025 12:34:56 +0900</pubDate>
          <description>
            <![CDATA[
              <p>Hello <b>world</b> &amp; friends. <a href="/more">More</a>.</p>
            ]]>
          </description>
        </item>
        <item>
          <title>Post Two</title>
          <link>https://example.com/post/2</link>
          <pubDate>Sun, 17 Aug 2025 10:00:00 +0000</pubDate>
          <description>Plain text</description>
        </item>
      </channel>
    </rss>
    """.strip()

    ingestor = KnueBoardIngestor()
    items = ingestor._parse_rss(xml)

    assert len(items) == 2
    assert items[0].title.startswith("Post One")
    assert items[0].link == "https://example.com/post/1"
    # Ensure tz-aware
    assert items[0].pub_date.tzinfo is not None
    # Description should be plain text without tags, whitespace collapsed
    assert items[0].description == "Hello world & friends. More."
    assert items[1].description == "Plain text"


@pytest.mark.unit
def test_parse_rss_handles_atom_like_variants():
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Example Atom</title>
      <entry>
        <title>Atom Post</title>
        <link href="https://example.com/atom/1" />
        <updated>2025-08-17T12:34:56+09:00</updated>
        <summary><![CDATA[<div>Summary <i>here</i></div>]]></summary>
      </entry>
    </feed>
    """.strip()

    ingestor = KnueBoardIngestor()
    items = ingestor._parse_rss(xml)

    assert len(items) == 1
    assert items[0].title == "Atom Post"
    assert items[0].link == "https://example.com/atom/1"
    # tz-aware
    assert items[0].pub_date.tzinfo is not None
    # Summary stripped to plain text
    assert items[0].description == "Summary here"
