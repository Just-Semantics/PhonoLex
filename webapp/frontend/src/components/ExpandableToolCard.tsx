/**
 * Expandable Tool Card Component
 *
 * A collapsible card that displays a tool with:
 * - Header with icon, title, and description
 * - Expandable content area
 * - Smooth animations
 */

import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Collapse,
  IconButton,
  Typography,
  Box,
  Avatar,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';

interface ExpandableToolCardProps {
  icon: React.ReactElement;
  title: string;
  description: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  color?: string;
}

const ExpandableToolCard: React.FC<ExpandableToolCardProps> = ({
  icon,
  title,
  description,
  children,
  defaultExpanded = false,
  color = 'primary.main',
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  return (
    <Card
      elevation={expanded ? 3 : 1}
      sx={{
        mb: { xs: 1.5, sm: 2 },
        transition: 'all 0.3s ease',
        '&:hover': {
          elevation: 2,
          boxShadow: 2,
        },
      }}
    >
      <CardHeader
        avatar={
          <Avatar
            sx={{
              bgcolor: color,
              width: { xs: 36, sm: 40 },
              height: { xs: 36, sm: 40 },
              '& .MuiSvgIcon-root': {
                fontSize: { xs: '1.25rem', sm: '1.5rem' },
              },
            }}
          >
            {icon}
          </Avatar>
        }
        action={
          <IconButton
            onClick={handleExpandClick}
            aria-expanded={expanded}
            aria-label="show more"
            sx={{
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s',
              minWidth: 44,
              minHeight: 44,
            }}
          >
            <ExpandMoreIcon />
          </IconButton>
        }
        title={
          <Typography
            variant="h6"
            component="div"
            fontWeight={600}
            sx={{
              fontSize: { xs: '1rem', sm: '1.25rem' },
              lineHeight: 1.3,
            }}
          >
            {title}
          </Typography>
        }
        subheader={
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              fontSize: { xs: '0.8125rem', sm: '0.875rem' },
              lineHeight: 1.4,
              pr: { xs: 1, sm: 0 },
            }}
          >
            {description}
          </Typography>
        }
        sx={{
          cursor: 'pointer',
          '&:hover': {
            bgcolor: 'action.hover',
          },
          py: { xs: 1.5, sm: 2 },
          px: { xs: 1.5, sm: 2 },
        }}
        onClick={handleExpandClick}
      />
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <CardContent
          sx={{
            px: { xs: 1.5, sm: 2 },
            py: { xs: 1.5, sm: 2 },
            '&:last-child': {
              pb: { xs: 2, sm: 2.5 },
            },
          }}
        >
          <Box sx={{ pt: { xs: 0, sm: 1 } }}>
            {children}
          </Box>
        </CardContent>
      </Collapse>
    </Card>
  );
};

export default ExpandableToolCard;
