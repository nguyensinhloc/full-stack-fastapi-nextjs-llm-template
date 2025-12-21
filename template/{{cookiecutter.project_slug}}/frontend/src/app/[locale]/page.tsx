import Link from "next/link";
import { Button, Card, CardHeader, CardTitle, CardContent } from "@/components/ui";
import { ROUTES } from "@/lib/constants";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-16 px-4">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">
            {{ cookiecutter.project_name }}
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            {{ cookiecutter.project_description }}
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 max-w-5xl mx-auto">
          {% if cookiecutter.use_jwt %}
          <Card>
            <CardHeader>
              <CardTitle>Authentication</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-muted-foreground">
                Secure JWT-based authentication system
              </p>
              <div className="flex gap-2">
                <Button asChild>
                  <Link href={ROUTES.LOGIN}>Login</Link>
                </Button>
                <Button variant="outline" asChild>
                  <Link href={ROUTES.REGISTER}>Register</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
          {% endif %}

          {% if cookiecutter.enable_ai_agent %}
          <Card>
            <CardHeader>
              <CardTitle>AI Assistant</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-muted-foreground">
                Chat with our AI assistant powered by PydanticAI
              </p>
              <Button asChild>
                <Link href={ROUTES.CHAT}>Start Chat</Link>
              </Button>
            </CardContent>
          </Card>
          {% endif %}

          <Card>
            <CardHeader>
              <CardTitle>Dashboard</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-muted-foreground">
                View your dashboard and manage your account
              </p>
              <Button variant="outline" asChild>
                <Link href={ROUTES.DASHBOARD}>Go to Dashboard</Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
