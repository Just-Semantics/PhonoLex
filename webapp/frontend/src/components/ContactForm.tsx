/**
 * Contact Form Component
 *
 * Uses Web3Forms API to send contact form submissions to email
 * without requiring a backend server.
 */

import React, { useState, FormEvent } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Stack,
} from '@mui/material';
import {
  Send as SendIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

interface FormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}

type SubmitStatus = 'idle' | 'submitting' | 'success' | 'error';

// TODO: Replace with your Web3Forms access key from https://web3forms.com/
// Sign up with contact@justsemantics.net to get your key
const WEB3FORMS_ACCESS_KEY = 'YOUR_WEB3FORMS_ACCESS_KEY_HERE';

export default function ContactForm() {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const [status, setStatus] = useState<SubmitStatus>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setStatus('submitting');
    setErrorMessage('');

    try {
      const formDataToSubmit = new FormData();
      formDataToSubmit.append('access_key', WEB3FORMS_ACCESS_KEY);
      formDataToSubmit.append('name', formData.name);
      formDataToSubmit.append('email', formData.email);
      formDataToSubmit.append('subject', formData.subject || `PhonoLex Contact: ${formData.name}`);
      formDataToSubmit.append('message', formData.message);
      formDataToSubmit.append('from_name', 'PhonoLex Contact Form');

      const response = await fetch('https://api.web3forms.com/submit', {
        method: 'POST',
        body: formDataToSubmit,
      });

      const result = await response.json();

      if (result.success) {
        setStatus('success');
        setFormData({ name: '', email: '', subject: '', message: '' });
      } else {
        setStatus('error');
        setErrorMessage('Failed to send message. Please try again or email us directly at contact@justsemantics.net');
      }
    } catch (error) {
      setStatus('error');
      setErrorMessage('An error occurred. Please try again or email contact@justsemantics.net');
      console.error('Form submission error:', error);
    }
  };

  if (status === 'success') {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Box
          sx={{
            width: 64,
            height: 64,
            bgcolor: 'success.light',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mx: 'auto',
            mb: 3,
          }}
        >
          <CheckCircleIcon sx={{ fontSize: 40, color: 'success.dark' }} />
        </Box>
        <Typography variant="h5" fontWeight={700} gutterBottom>
          Message Sent!
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Thank you for reaching out. We'll get back to you within 24-48 hours.
        </Typography>
        <Button
          variant="outlined"
          onClick={() => setStatus('idle')}
          sx={{ mt: 2 }}
        >
          Send Another Message
        </Button>
      </Box>
    );
  }

  return (
    <Box component="form" onSubmit={handleSubmit} noValidate>
      <Typography variant="h6" fontWeight={600} gutterBottom>
        Send Us a Message
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Have questions, feedback, or collaboration ideas? We'd love to hear from you.
      </Typography>

      {status === 'error' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {errorMessage}
        </Alert>
      )}

      <Stack spacing={3}>
        {/* Name Field */}
        <TextField
          fullWidth
          label="Name"
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
          placeholder="Your full name"
          disabled={status === 'submitting'}
        />

        {/* Email Field */}
        <TextField
          fullWidth
          type="email"
          label="Email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          required
          placeholder="your.email@example.com"
          disabled={status === 'submitting'}
        />

        {/* Subject Field */}
        <TextField
          fullWidth
          label="Subject"
          name="subject"
          value={formData.subject}
          onChange={handleChange}
          placeholder="What's this about?"
          disabled={status === 'submitting'}
        />

        {/* Message Field */}
        <TextField
          fullWidth
          multiline
          rows={6}
          label="Message"
          name="message"
          value={formData.message}
          onChange={handleChange}
          required
          placeholder="Tell us about your question, feedback, or collaboration idea..."
          disabled={status === 'submitting'}
        />

        {/* Submit Button */}
        <Button
          type="submit"
          variant="contained"
          size="large"
          disabled={status === 'submitting'}
          startIcon={status === 'submitting' ? <CircularProgress size={20} /> : <SendIcon />}
          fullWidth
          sx={{ py: 1.5 }}
        >
          {status === 'submitting' ? 'Sending...' : 'Send Message'}
        </Button>

        <Typography variant="caption" color="text.secondary" align="center">
          We typically respond within 24-48 hours
        </Typography>
      </Stack>
    </Box>
  );
}
