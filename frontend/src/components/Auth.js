import React from 'react';
import { GoogleLogin } from '@react-oauth/google';
import axios from 'axios';

function Auth({ setUser }) {
  const onSuccess = async (response) => {
    try {
      const res = await axios.post('https://legaldocs-ai.onrender.com/api/auth/google', { token: response.credential });
      setUser({ token: response.credential, ...res.data });
    } catch (error) {
      console.error('Auth error:', error);
    }
  };

  return (
    <GoogleLogin
      onSuccess={onSuccess}
      onError={() => console.log('Login Failed')}
    />
  );
}

export default Auth;