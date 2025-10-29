/**
 * Privacy Policy Page
 */

import React from 'react';
import { Container, Typography, Box, Link, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

const PrivacyPolicy: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/')}
        sx={{ mb: 3 }}
      >
        Back to App
      </Button>

      <Typography variant="h3" component="h1" gutterBottom fontWeight={600}>
        Privacy Policy
      </Typography>

      <Typography variant="body2" color="text.secondary" paragraph>
        Last updated: {new Date().toLocaleDateString()}
      </Typography>

      <Box sx={{ '& > *': { mb: 3 } }}>
        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Overview
          </Typography>
          <Typography variant="body1" paragraph>
            PhonoLex is committed to protecting your privacy. This policy explains how we collect, use, and safeguard your information when you use our web application.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Information We Collect
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>No Personal Data Collection:</strong> PhonoLex does not collect, store, or transmit any personally identifiable information (PII). We do not require user accounts, login credentials, or any personal details.
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>Usage Data:</strong> We may collect anonymous, aggregated usage statistics through standard web server logs, including:
          </Typography>
          <Typography variant="body1" component="ul" sx={{ pl: 4 }}>
            <li>Browser type and version</li>
            <li>Operating system</li>
            <li>Page views and navigation patterns</li>
            <li>Time spent on pages</li>
            <li>Referring URLs</li>
          </Typography>
          <Typography variant="body1" paragraph>
            This data is used solely for improving the application and is never linked to individual users.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Cookies and Local Storage
          </Typography>
          <Typography variant="body1" paragraph>
            PhonoLex may use local browser storage to save user preferences (e.g., theme settings, tool configurations). This data remains entirely on your device and is never transmitted to our servers.
          </Typography>
          <Typography variant="body1" paragraph>
            We do not use tracking cookies or third-party analytics services.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Data Security
          </Typography>
          <Typography variant="body1" paragraph>
            All queries and interactions with PhonoLex occur over secure HTTPS connections. Since we do not collect personal data, there is no risk of personal information exposure.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Third-Party Services
          </Typography>
          <Typography variant="body1" paragraph>
            PhonoLex does not integrate with third-party services that collect user data. All data processing occurs on our own infrastructure.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Children's Privacy
          </Typography>
          <Typography variant="body1" paragraph>
            PhonoLex does not knowingly collect information from children under 13. The application is designed for educational and research purposes and does not require age verification.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Changes to This Policy
          </Typography>
          <Typography variant="body1" paragraph>
            We may update this Privacy Policy periodically. Changes will be posted on this page with an updated revision date.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Contact
          </Typography>
          <Typography variant="body1" paragraph>
            For questions about this Privacy Policy, please contact{' '}
            <Link href="mailto:contact@justsemantics.net" underline="hover">
              contact@justsemantics.net
            </Link>.
          </Typography>
        </Box>
      </Box>
    </Container>
  );
};

export default PrivacyPolicy;
