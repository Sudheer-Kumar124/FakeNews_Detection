import streamlit as st

def show_statistics():
    st.markdown("<h2>Statistics</h2>", unsafe_allow_html=True)
    
    # Example statistic 1
    st.markdown("<h3>Example Statistic 1</h3>", unsafe_allow_html=True)
    st.image("path/to/image1.jpg", caption="Description of the first statistic", use_column_width=True)
    st.markdown("""
    This is a brief description of the first statistic.
    [Link to more information](https://example.com/statistic1)
    """, unsafe_allow_html=True)

    # Example statistic 2
    st.markdown("<h3>Example Statistic 2</h3>", unsafe_allow_html=True)
    st.image("path/to/image2.jpg", caption="Description of the second statistic", use_column_width=True)
    st.markdown("""
    This is a brief description of the second statistic.
    [Link to more information](https://example.com/statistic2)
    """, unsafe_allow_html=True)

    # Example statistic 3
    st.markdown("<h3>Example Statistic 3</h3>", unsafe_allow_html=True)
    st.image("path/to/image3.jpg", caption="Description of the third statistic", use_column_width=True)
    st.markdown("""
    This is a brief description of the third statistic.
    [Link to more information](https://example.com/statistic3)
    """, unsafe_allow_html=True)

# Entry point
if __name__ == "__main__":
    show_statistics()
