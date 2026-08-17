import streamlit as st


USERS = {
    "admin": {
        "password": "admin123",
        "role": "Admin"
    },
    "customer": {
        "password": "customer123",
        "role": "Customer"
    }
}


def login():

    st.sidebar.title("ReviewSense Login")

    username = st.sidebar.text_input(
        "Username"
    )

    password = st.sidebar.text_input(
        "Password",
        type="password"
    )

    if st.sidebar.button("Login"):

        if username in USERS:

            if USERS[username]["password"] == password:

                st.session_state["authenticated"] = True

                st.session_state["username"] = username

                st.session_state["role"] = (
                    USERS[username]["role"]
                )

                st.rerun()

            else:

                st.sidebar.error(
                    "Invalid password"
                )

        else:

            st.sidebar.error(
                "User not found"
            )


def logout():

    if st.sidebar.button("Logout"):

        for key in [
            "authenticated",
            "username",
            "role"
        ]:

            st.session_state.pop(
                key,
                None
            )

        st.rerun()


def is_authenticated():

    return st.session_state.get(
        "authenticated",
        False
    )


def get_role():

    return st.session_state.get(
        "role",
        None
    )
