/**
 * Forge app: AI Sprint Assistant
 * Adds a button to Jira sprint board that calls LLM and posts summary comment.
 */
import ForgeUI, { render, Button, Text, useProductContext, useState } from '@forge/ui';
import { api } from '@forge/api';

const App = () => {
  const context = useProductContext();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyzeSprint = async () => {
    setLoading(true);
    setResult(null);

    try {
      // Get current sprint issues
      const issues = await api.asApp().requestJira('/rest/api/3/search', {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        },
        params: {
          jql: `sprint in openSprints() AND project = ${context.platformContext.extension.projectKey}`
        }
      });

      const issuesData = await issues.json();
      
      // Summarize with LLM (call external API or use Forge's built-in)
      const summary = await generateSummary(issuesData.issues);

      // Post comment to sprint
      await postComment(summary);

      setResult(`Sprint summary posted! Analyzed ${issuesData.issues.length} issues.`);
    } catch (error) {
      setResult(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ForgeUI>
      <Text>AI Sprint Assistant</Text>
      <Button 
        text={loading ? "Analyzing..." : "Analyze Sprint"} 
        onClick={handleAnalyzeSprint}
        disabled={loading}
      />
      {result && <Text>{result}</Text>}
    </ForgeUI>
  );
};

async function generateSummary(issues) {
  // Call your LLM API (OpenAI, Anthropic, etc.)
  // For demo, return placeholder
  const issueTexts = issues.map(i => `${i.key}: ${i.fields.summary}`).join('\n');
  
  return `Sprint Summary:\n- ${issues.length} issues\n- Key items: ${issueTexts.substring(0, 200)}`;
}

async function postComment(summary) {
  // Get sprint board ID from context
  // Post comment via Jira API
  // Implementation depends on exact sprint/board structure
}

export const run = render(<App />);


