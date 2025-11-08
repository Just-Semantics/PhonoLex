import React from 'react';
import { Box, Container, Typography, Paper } from '@mui/material';
import { theme } from '../theme/theme';

interface AspectRatioPreview {
  name: string;
  ratio: number; // width / height
  width: number;
  height: number;
  description: string;
}

const ASPECT_RATIOS: AspectRatioPreview[] = [
  {
    name: 'Facebook Open Graph',
    ratio: 1.91,
    width: 1200,
    height: 630,
    description: 'Recommended: 1200x630px (1.91:1)',
  },
  {
    name: 'Twitter Card Large',
    ratio: 2.0,
    width: 1200,
    height: 600,
    description: 'Summary Card with Large Image: 1200x600px (2:1)',
  },
  {
    name: 'LinkedIn',
    ratio: 1.91,
    width: 1200,
    height: 627,
    description: 'Recommended: 1200x627px (1.91:1)',
  },
  {
    name: 'Twitter Card Square',
    ratio: 1.0,
    width: 1200,
    height: 1200,
    description: 'Summary Card: 1200x1200px (1:1)',
  },
  {
    name: 'Instagram',
    ratio: 1.0,
    width: 1080,
    height: 1080,
    description: 'Square Post: 1080x1080px (1:1)',
  },
];

const LogoPreview: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom fontWeight={600} color="primary">
        PhonoLex Logo Preview
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Logo displayed at different social media aspect ratios on the app header background color ({theme.palette.primary.main})
      </Typography>

      <Box sx={{ mt: 2 }}>
        {ASPECT_RATIOS.map((ar) => (
          <Box key={ar.name} sx={{ mb: 6 }}>
            <Typography variant="h6" gutterBottom fontWeight={600}>
              {ar.name}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {ar.description}
            </Typography>

            {/* Preview container with EXTRA background for easier cropping */}
            <Box
              sx={{
                mt: 2,
                width: `${ar.width + 100}px`,
                height: `${ar.height + 100}px`,
                bgcolor: theme.palette.primary.main,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {/* Logo - clean for screenshots */}
              <Box
                component="img"
                src="/logo.png"
                alt="PhonoLex Logo"
                sx={{
                  maxWidth: '60%',
                  maxHeight: '60%',
                  objectFit: 'contain',
                  filter: 'brightness(0) invert(1)', // Make logo white
                }}
              />
            </Box>

            {/* Dimensions info below image */}
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', fontWeight: 600 }}>
              Screenshot this: {ar.width} × {ar.height}px (aspect ratio {ar.ratio}:1)
            </Typography>
          </Box>
        ))}
      </Box>

      {/* Instructions */}
      <Paper elevation={1} sx={{ mt: 4, p: 3, bgcolor: 'primary.50' }}>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          Usage Instructions
        </Typography>
        <Typography variant="body2" component="div">
          <ol style={{ paddingLeft: 20 }}>
            <li>Right-click on any preview above to inspect the dimensions</li>
            <li>Take a screenshot to create the actual social media image</li>
            <li>For production, use design software to create exact dimensions</li>
            <li>Logo filter: <code>brightness(0) invert(1)</code> makes it white</li>
            <li>Background color: <code>{theme.palette.primary.main}</code></li>
          </ol>
        </Typography>
      </Paper>
    </Container>
  );
};

export default LogoPreview;
