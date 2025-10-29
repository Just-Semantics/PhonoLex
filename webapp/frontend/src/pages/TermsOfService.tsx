/**
 * Terms of Service Page
 */

import React from 'react';
import { Container, Typography, Box, Link, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

const TermsOfService: React.FC = () => {
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
        Terms of Service
      </Typography>

      <Typography variant="body2" color="text.secondary" paragraph>
        Last updated: {new Date().toLocaleDateString()}
      </Typography>

      <Box sx={{ '& > *': { mb: 3 } }}>
        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Acceptance of Terms
          </Typography>
          <Typography variant="body1" paragraph>
            By accessing and using PhonoLex ("the Service"), you accept and agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the Service.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Description of Service
          </Typography>
          <Typography variant="body1" paragraph>
            PhonoLex is an open-source phonological analysis tool providing word filtering, minimal pair generation, rhyme detection, and similarity computation. The Service is provided free of charge for educational and research purposes.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            No Warranty
          </Typography>
          <Typography variant="body1" paragraph>
            THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.
          </Typography>
          <Typography variant="body1" paragraph>
            While reasonable efforts have been made to ensure accuracy, PhonoLex makes no guarantees regarding:
          </Typography>
          <Typography variant="body1" component="ul" sx={{ pl: 4 }}>
            <li>Correctness or completeness of phonological data</li>
            <li>Accuracy of linguistic annotations or embeddings</li>
            <li>Suitability for clinical, therapeutic, or diagnostic purposes</li>
            <li>Continuous availability or uptime of the Service</li>
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Limitation of Liability
          </Typography>
          <Typography variant="body1" paragraph>
            IN NO EVENT SHALL JUST SEMANTICS, PHONOLEX, ITS CONTRIBUTORS, OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SERVICE.
          </Typography>
          <Typography variant="body1" paragraph>
            Users assume full responsibility for verifying results and determining appropriateness for their specific use cases.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Acceptable Use
          </Typography>
          <Typography variant="body1" paragraph>
            You agree to use the Service only for lawful purposes. You may not:
          </Typography>
          <Typography variant="body1" component="ul" sx={{ pl: 4 }}>
            <li>Attempt to disrupt or compromise the Service's infrastructure</li>
            <li>Use automated tools to overload or abuse the Service</li>
            <li>Misrepresent results or data obtained from the Service</li>
            <li>Use the Service for any illegal or unauthorized purpose</li>
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Intellectual Property
          </Typography>
          <Typography variant="body1" paragraph>
            PhonoLex is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA 3.0) due to incorporated data sources. Source code is available on{' '}
            <Link href="https://github.com/neumanns-workshop/PhonoLex" target="_blank" rel="noopener noreferrer">
              GitHub
            </Link>.
          </Typography>
          <Typography variant="body1" paragraph>
            This ShareAlike license is required because we incorporate PHOIBLE phonological data (CC BY-SA 3.0).
            Any derivative works must also be shared under CC BY-SA 3.0 or a compatible license.
          </Typography>
          <Typography variant="body1" paragraph>
            See the LICENSE file and Info section for complete data source citations and attribution requirements.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Research and Academic Use
          </Typography>
          <Typography variant="body1" paragraph>
            If you use PhonoLex in academic research or publications, we encourage (but do not require) citation of the underlying datasets and methods. See the Info section for complete bibliography.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Modifications to Service
          </Typography>
          <Typography variant="body1" paragraph>
            We reserve the right to modify, suspend, or discontinue the Service at any time without notice or liability.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Changes to Terms
          </Typography>
          <Typography variant="body1" paragraph>
            These Terms may be updated periodically. Continued use of the Service constitutes acceptance of revised Terms.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Governing Law
          </Typography>
          <Typography variant="body1" paragraph>
            These Terms shall be governed by and construed in accordance with applicable laws, without regard to conflict of law provisions.
          </Typography>
        </Box>

        <Box>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Contact
          </Typography>
          <Typography variant="body1" paragraph>
            For questions about these Terms, please visit our{' '}
            <Link href="https://github.com/neumanns-workshop/PhonoLex" target="_blank" rel="noopener noreferrer">
              GitHub repository
            </Link>.
          </Typography>
        </Box>
      </Box>
    </Container>
  );
};

export default TermsOfService;
